"""Single-codebook VQ4DiT, equations5-11; no additive or residual codebooks."""
import torch
from torch import nn
from torch.nn import functional as F


@torch.no_grad()
def nearest_candidates(vectors, book, count=2, chunk=32768):
    if vectors.ndim != 2 or book.ndim != 2 or vectors.shape[1] != book.shape[1]:
        raise ValueError('Incompatible vectors/codebook')
    if not 1 <= count <= len(book) <= 256 or chunk < 1:
        raise ValueError('Expected up to256 centroids and valid chunk/count')
    result = torch.empty((len(vectors), count), dtype=torch.uint8, device=vectors.device)
    for start in range(0, len(vectors), chunk):
        rows = vectors[start:start+chunk]
        distances = (rows[:, None, :]-book[None, :, :]).square().sum(-1)
        result[start:start+len(rows)] = distances.topk(count, dim=1, largest=False, sorted=True).indices.to(torch.uint8)
    return result


class PaperLinear(nn.Module):
    def __init__(self, book, candidates, shape, bias=None):
        super().__init__()
        if candidates.ndim != 2 or candidates.shape[1] != 2 or candidates.dtype != torch.uint8:
            raise ValueError('Exactly two uint8 candidate indices are required')
        if candidates.shape[0]*book.shape[1] != shape[0]*shape[1] or shape[1] % book.shape[1]:
            raise ValueError('Weight shape does not match its vectors')
        if torch.any(candidates[:, 0] == candidates[:, 1]) or int(candidates.max()) >= len(book):
            raise ValueError('Candidates must be distinct valid codewords')
        self.shape = tuple(shape)
        self.book = nn.Parameter(book.detach().clone().float())
        self.register_buffer('candidates', candidates.detach().clone())
        self.logits = nn.Parameter(torch.zeros(candidates.shape, device=book.device, dtype=torch.float32))
        self.bias = nn.Parameter(bias.detach().clone().float()) if bias is not None else None
        self.hardened = False

    def weight(self):
        if self.hardened:
            values = self.book[self.hard_codes.long()]
        else:
            probabilities = self.logits.softmax(-1)
            values = (self.book[self.candidates.long()]*probabilities[..., None]).sum(1)
        return values.reshape(self.shape)

    def polarization(self):
        if self.hardened:
            return self.book.new_zeros(())
        probabilities = self.logits.softmax(-1)
        # Equation9 averages over vectors, not over candidate count.
        return (1-(2*probabilities-1).abs()).sum()/len(probabilities)

    @torch.no_grad()
    def codes(self):
        if self.hardened:
            return self.hard_codes.clone()
        winner = self.logits.argmax(-1, keepdim=True)
        return self.candidates.gather(1, winner).squeeze(1)

    @torch.no_grad()
    def harden(self):
        if self.hardened:
            return
        self.register_buffer('hard_codes', self.codes())
        self.logits.requires_grad_(False)
        self.hardened = True

    @torch.no_grad()
    def exported(self):
        return dict(book=self.book.detach().cpu(), codes=self.codes().cpu(), shape=self.shape,
                    bias=self.bias.detach().cpu() if self.bias is not None else None)

    def forward(self, x):
        return F.linear(x, self.weight(), self.bias)


def block_loss(outputs, targets):
    if len(outputs) != len(targets) or not outputs:
        raise ValueError('Missing block targets')
    return torch.stack([F.mse_loss(x, y) for x, y in zip(outputs, targets)]).sum()


def storage(layers, unquantized_parameters):
    target = sum(m.shape[0]*m.shape[1] for _, m in layers)
    bits = sum(len(m.candidates)*8 + m.book.numel()*32
               + (m.bias.numel()*32 if m.bias is not None else 0) for _, m in layers)
    total_parameters = target+unquantized_parameters+sum(m.bias.numel() for _, m in layers if m.bias is not None)
    return dict(logical_bits=bits+unquantized_parameters*32,
                whole_dit_bpp=(bits+unquantized_parameters*32)/total_parameters,
                whole_dit_MB=(bits+unquantized_parameters*32)/8e6,
                quantized_block_bits=bits, quantized_block_MiB=bits/8/1024**2,
                target_weights=target, non_block_parameters=unquantized_parameters,
                nominal_index_bits=2, codebook_dtype='float32', index_dtype='uint8',
                note='Includes bias and frozen non-block parameters; no logits, candidates, optimizer or reference weights.')
