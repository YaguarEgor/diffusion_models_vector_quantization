"""Additive weights and loss-guided discrete updates; independent of cluster code."""

import contextlib
import math

import torch
from torch import nn
from torch.nn import functional as F

LAYERS = ("attn.qkv", "attn.proj", "mlp.fc1", "mlp.fc2", "adaLN_modulation.1")
OBJECTIVES = ("blocks", "teacher", "noise")


def dequantize(books, codes, scales):
    # Same shared-codebook embedding_bag layout and arithmetic as native VQDM.
    if books.ndim != 5 or books.shape[0] != 1 or books.shape[-2] != 1:
        raise ValueError("Expected shared additive books [1,M,K,1,d]")
    _, m, k, _, d = books.shape
    o, g, stages = codes.shape
    if stages != m or scales.shape != (o, 1, 1, 1):
        raise ValueError("Invalid codes/scales shape")
    offsets = torch.arange(m, device=codes.device) * k
    values = F.embedding_bag((codes.long() + offsets).reshape(-1, m),
                             books.reshape(m * k, d), mode="sum")
    return (values.reshape(o, g, 1, d) * scales).swapaxes(1, 2).reshape(o, g * d)


class AdditiveLinear(nn.Module):
    def __init__(self, books, codes, scales, bias):
        super().__init__()
        self.books = nn.Parameter(books.detach().float().clone())
        self.register_buffer("codes", codes.detach().to(torch.int32).clone())
        self.register_buffer("scales", scales.detach().half().clone())
        self.register_buffer("bias", None if bias is None else bias.detach().half().clone())
        self.capture = False
        self.captured = None

    def weight(self):
        # Gradients reach FP32 master books through the FP16 cast. Every forward
        # uses the actual stored FP16 reconstruction, not a higher precision proxy.
        return dequantize(self.books.half(), self.codes, self.scales).float()

    def forward(self, x):
        weight = self.weight()
        if self.capture:
            weight = weight.detach().requires_grad_(True)
            self.captured = weight
        return F.linear(x, weight, None if self.bias is None else self.bias.float())


def loss_terms(prediction, teacher, noise, student_blocks, teacher_blocks, objective,
               variance_weight=.1):
    if objective not in OBJECTIVES or prediction.shape[1] != 2 * noise.shape[1]:
        raise ValueError("Invalid objective or DiT epsilon/variance output")
    eps, var = prediction.chunk(2, dim=1)
    teps, tvar = teacher.chunk(2, dim=1)
    teacher_loss = F.mse_loss(eps, teps)
    noise_loss = F.mse_loss(eps, noise)
    variance = F.mse_loss(var, tvar)
    if objective == "blocks":
        if not student_blocks or len(student_blocks) != len(teacher_blocks):
            raise ValueError("Missing matched block outputs")
        main = torch.stack([F.mse_loss(s, t) for s, t in zip(student_blocks, teacher_blocks)]).mean()
    else:
        main = teacher_loss if objective == "teacher" else noise_loss
    total = main + variance_weight * variance
    return total, dict(main=main, epsilon_teacher=teacher_loss, epsilon_noise=noise_loss,
                      variance_teacher=variance)


@contextlib.contextmanager
def block_outputs(model, enabled):
    values, handles = [], []
    if enabled:
        for block in model.blocks:
            handles.append(block.register_forward_hook(lambda module, args, output: values.append(output)))
    try:
        yield values
    finally:
        for handle in handles:
            handle.remove()


@torch.no_grad()
def propose(layer, weight_gradient, candidate_groups=2048, changes=128, trust=.25, chunk=128):
    """One stage replacement per input group; quadratic scores are only proposals."""
    if not 0 < candidate_groups or not 0 < changes or trust <= 0:
        raise ValueError("Invalid proposal limits")
    books = layer.books.detach().half().float()[0, :, :, 0]
    m, k, d = books.shape
    o, g, _ = layer.codes.shape
    if weight_gradient.shape != (o, g * d) or not torch.isfinite(weight_gradient).all():
        raise ValueError("Invalid dense weight gradient")
    grad = weight_gradient.reshape(o * g, d)
    # Stable ordering makes tied zero-gradient groups reproducible.
    ids = torch.argsort(grad.square().sum(-1), descending=True, stable=True)[:candidate_groups]
    flat_codes = layer.codes.reshape(-1, m)
    scale = layer.scales.reshape(o).float()
    results = []
    for start in range(0, len(ids), chunk):
        selected = ids[start:start + chunk]
        h = grad[selected]
        s = scale[selected // g]
        best_score = torch.zeros(len(selected), device=h.device)
        best_stage = torch.zeros(len(selected), dtype=torch.long, device=h.device)
        best_code = flat_codes[selected, 0].long().clone()
        for stage in range(m):
            current = books[stage, flat_codes[selected, stage].long()]
            delta = (books[stage][None] - current[:, None]) * s[:, None, None]
            # Scale the trust coefficient to this probe's gradient and code steps.
            curvature = trust * h.norm(dim=-1) / (delta.square().sum(-1).mean(-1).sqrt() + 1e-12)
            score = (delta * h[:, None]).sum(-1) + .5 * curvature[:, None] * delta.square().sum(-1)
            value, code = score.min(-1)
            improve = value < best_score
            best_score = torch.where(improve, value, best_score)
            best_stage = torch.where(improve, stage, best_stage)
            best_code = torch.where(improve, code, best_code)
        for group, stage, code, score in zip(selected.tolist(), best_stage.tolist(),
                                            best_code.tolist(), best_score.tolist()):
            if score < 0:
                results.append((group, stage, code, score))
    return sorted(results, key=lambda row: (row[3], row[0], row[1]))[:changes]


def accept_proposal(layer, proposal, evaluate, minimum_gain=1e-9):
    """Accept only a measured improvement of the same deterministic true loss."""
    before = float(evaluate())
    if not math.isfinite(before):
        raise ValueError("Nonfinite pre-recode loss")
    codes = layer.codes.reshape(-1, layer.codes.shape[-1])
    original = [(group, stage, int(codes[group, stage])) for group, stage, _, _ in proposal]
    if len({group for group, _, _, _ in proposal}) != len(proposal):
        raise ValueError("At most one proposal per group is allowed")
    attempts = []
    size = len(proposal)
    try:
        while size:
            with torch.no_grad():
                for group, stage, old in original:
                    codes[group, stage] = old
                for group, stage, code, _ in proposal[:size]:
                    codes[group, stage] = code
            after = float(evaluate())
            attempts.append(dict(changes=size, loss=after if math.isfinite(after) else None))
            if math.isfinite(after) and after < before - minimum_gain:
                return dict(before=before, after=after, accepted=size, attempts=attempts)
            size //= 2
    except BaseException:
        with torch.no_grad():
            for group, stage, old in original:
                codes[group, stage] = old
        raise
    with torch.no_grad():
        for group, stage, old in original:
            codes[group, stage] = old
    return dict(before=before, after=before, accepted=0, attempts=attempts)
