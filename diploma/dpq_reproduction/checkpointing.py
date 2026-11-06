"""Infrastructure-independent checkpoint wrapper extracted from the DPQ runner."""

import contextlib

import torch
from torch.utils.checkpoint import checkpoint

from paper_pq import PaperPQLinear


def pq_modules(model, mode):
    from pqf.compressed_layers.CompressedLinear import CompressedLinear
    kind = CompressedLinear if mode == 'upstream' else PaperPQLinear
    return [(name, layer) for name, layer in model.named_modules() if isinstance(layer, kind)]


def attach_checkpointing(model, mode):
    @contextlib.contextmanager
    def no_assignments():
        selected = pq_modules(model, mode) if mode == 'paper' else []
        flags = [layer.e_enabled for _, layer in selected]
        try:
            for _, layer in selected:
                layer.e_enabled = False
            yield
        finally:
            for (_, layer), flag in zip(selected, flags):
                layer.e_enabled = flag
    for block in model.blocks:
        original = block.forward
        def forward(x, c, original=original):
            if torch.is_grad_enabled():
                return checkpoint(original, x, c, use_reentrant=False,
                                  context_fn=lambda: (contextlib.nullcontext(), no_assignments()))
            return original(x, c)
        block.forward = forward
