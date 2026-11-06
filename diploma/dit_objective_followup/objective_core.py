"""Training-only extensions of the pinned additive representation."""
import contextlib
import math

import torch
from torch import nn
from torch.nn import functional as F

from core import AdditiveLinear, dequantize


class FollowupLinear(AdditiveLinear):
    def __init__(self, values, train_scales=False, alternatives=None):
        super().__init__(values['books'], values['codes'], values['scales'], values['bias'])
        self.log_scale = nn.Parameter(torch.zeros_like(self.scales, dtype=torch.float32)) if train_scales else None
        self.register_buffer('alternatives', alternatives)
        self.logits = nn.Parameter(torch.full_like(self.codes, math.log(.1 / .9), dtype=torch.float32)) if alternatives is not None else None
        self.hard = False

    def effective_scales(self):
        if self.log_scale is None:
            return self.scales
        return (self.scales.float() * self.log_scale.exp()).half()

    def hard_codes(self):
        if self.logits is None:
            return self.codes
        return torch.where(self.logits > 0, self.alternatives, self.codes)

    def weight(self):
        if self.logits is None or self.hard:
            return dequantize(self.books.half(), self.hard_codes(), self.effective_scales()).float()
        _, m, k, _, d = self.books.shape
        o, g, _ = self.codes.shape
        offsets = torch.arange(m, device=self.codes.device) * k
        indices = torch.stack((self.codes.long() + offsets, self.alternatives.long() + offsets), -1)
        p = self.logits.sigmoid()
        # Keep the relaxation FP32: per-assignment probability gradients can be
        # much smaller than the FP16 range. Codebook VALUES still round to FP16.
        mixing = torch.stack((1 - p, p), -1)
        value = F.embedding_bag(indices.reshape(-1, 2*m), self.books.half().float().reshape(m*k, d),
                                per_sample_weights=mixing.reshape(-1, 2*m), mode='sum')
        return (value.reshape(o, g, 1, d) * self.effective_scales().float()).swapaxes(1, 2).reshape(o, g*d)

    def exported(self):
        return dict(books=self.books.detach().half().cpu(), codes=self.hard_codes().detach().byte().cpu(),
                    scales=self.effective_scales().detach().cpu(),
                    bias=None if self.bias is None else self.bias.detach().cpu())


@contextlib.contextmanager
def hard_assignments(model):
    modules = [m for m in model.modules() if isinstance(m, FollowupLinear)]
    previous = [m.hard for m in modules]
    try:
        for module in modules:
            module.hard = True
        yield
    finally:
        for module, value in zip(modules, previous):
            module.hard = value


def polarization(model):
    values = [(m.logits, m.logits.numel()) for m in model.modules()
              if isinstance(m, FollowupLinear) and m.logits is not None]
    if not values:
        return next(model.parameters()).new_zeros(())
    total = sum(n for _, n in values)
    # Mean per assignment, including the two binary candidate probabilities.
    return sum(2 * (1 - (2*p.sigmoid()-1).abs()).mean() * (n/total) for p,n in values)


@torch.no_grad()
def nearest_alternatives(values, reference, chunk=16384):
    books = values['books'].float()[0, :, :, 0]
    codes = values['codes'].long()
    o, g, m = codes.shape
    if books.shape[0] != m or reference.shape != (o, g*books.shape[-1]):
        raise ValueError('Invalid additive reference dimensions')
    d = books.shape[-1]
    vectors = reference.float().reshape(o, g, d) / values['scales'].float().reshape(o, 1, 1)
    summed = sum(F.embedding(codes[..., s], books[s]) for s in range(m))
    result = torch.empty_like(codes, dtype=torch.int32)
    for stage in range(m):
        target = (vectors - summed + F.embedding(codes[..., stage], books[stage])).reshape(-1, d)
        current = codes[..., stage].reshape(-1)
        selected = []
        centers = books[stage]
        norms = centers.square().sum(1)
        for start in range(0, len(target), chunk):
            x = target[start:start+chunk]
            distances = x.square().sum(1, keepdim=True) + norms - 2*x @ centers.T
            distances.scatter_(1, current[start:start+chunk, None], float('inf'))
            selected.append(distances.argmin(1).int())
        result[..., stage] = torch.cat(selected).reshape(o, g)
    if torch.any(result == codes):
        raise ValueError('Alternative candidates must differ from the initial indices')
    return result


def ddpm_objective(diffusion, prediction, clean, noisy, timesteps, noise):
    eps, variance = prediction.chunk(2, dim=1)
    frozen = torch.cat((eps.detach(), variance), dim=1)
    vb = diffusion._vb_terms_bpd(lambda *args, **kwargs: frozen, x_start=clean, x_t=noisy,
                               t=timesteps, clip_denoised=False)['output']
    mse = (noise-eps).square().mean(dim=tuple(range(1,eps.ndim)))
    # Match upstream's per-image addition before averaging, not two global means.
    return (mse+vb).mean(), dict(epsilon_noise=mse.mean(), variance_vb=vb.mean())
