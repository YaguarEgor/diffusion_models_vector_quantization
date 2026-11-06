import copy
import unittest

import torch
from torch import nn

from core import AdditiveLinear, accept_proposal, block_outputs, dequantize, loss_terms, propose


class Tests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(5)
        self.layer = AdditiveLinear(torch.randn(1, 2, 4, 1, 2) * .1,
                                    torch.randint(4, (3, 2, 2)), torch.ones(3, 1, 1, 1),
                                    torch.zeros(3))

    def test_layout_and_gradient(self):
        layer = self.layer
        reference = torch.stack([torch.cat([sum(layer.books[0, m, layer.codes[o, g, m], 0]
                                                for m in range(2)) for g in range(2)])
                                 for o in range(3)])
        self.assertTrue(torch.allclose(dequantize(layer.books, layer.codes, layer.scales.float()), reference))
        codes = layer.codes.clone()
        layer(torch.randn(2, 4)).square().sum().backward()
        self.assertGreater(layer.books.grad.abs().sum().item(), 0)
        self.assertTrue(torch.equal(codes, layer.codes))

    def test_true_loss_and_rollback(self):
        layer = self.layer
        old = layer.codes.clone()
        proposed = [(0, 0, (int(old[0, 0, 0]) + 1) % 4, -1.)]
        result = accept_proposal(layer, proposed, lambda: float((layer.codes != old).sum()))
        self.assertEqual(result['accepted'], 0)
        self.assertTrue(torch.equal(old, layer.codes))
        result = accept_proposal(layer, proposed, lambda: -float((layer.codes != old).sum()))
        self.assertEqual(result['accepted'], 1)
        self.assertLess(result['after'], result['before'])

    def test_exception_rolls_back(self):
        old = self.layer.codes.clone()
        calls = []
        def fail():
            calls.append(1)
            if len(calls) > 1:
                raise RuntimeError('probe failed')
            return 1
        with self.assertRaises(RuntimeError):
            accept_proposal(self.layer, [(0, 0, (int(old[0, 0, 0]) + 1) % 4, -1)], fail)
        self.assertTrue(torch.equal(old, self.layer.codes))

    def test_proposal_improves_linear_objective(self):
        layer = self.layer
        grad = torch.randn(3, 4)
        proposal = propose(layer, grad, candidate_groups=6, changes=6)
        before = layer.weight().detach().clone()
        result = accept_proposal(layer, proposal, lambda: (layer.weight() * grad).sum().item())
        self.assertGreater(result['accepted'], 0)
        self.assertLess(((layer.weight() - before) * grad).sum().item(), 0)
        self.assertEqual(len({row[0] for row in proposal}), len(proposal))

    def test_objectives_and_variance(self):
        noise = torch.randn(2, 4, 2, 2)
        teacher = torch.randn(2, 8, 2, 2)
        prediction = teacher.clone().requires_grad_()
        blocks = [torch.zeros(2, 3, requires_grad=True)]
        for objective in ('blocks', 'teacher', 'noise'):
            loss, terms = loss_terms(prediction, teacher, noise, blocks, [torch.ones(2, 3)], objective)
            self.assertEqual(float(terms['variance_teacher'].detach()), 0)
            self.assertAlmostEqual(float(loss.detach()), 1 if objective == 'blocks' else
                                   0 if objective == 'teacher' else float(terms['epsilon_noise'].detach()))

    def test_hook_cleanup_and_capture(self):
        model = nn.Module()
        model.blocks = nn.ModuleList([nn.Linear(4, 4), nn.Linear(4, 4)])
        with block_outputs(model, True) as values:
            x = torch.randn(1, 4)
            for block in model.blocks:
                x = block(x)
            self.assertEqual(len(values), 2)
        self.assertFalse(model.blocks[0]._forward_hooks)
        self.layer.capture = True
        self.layer(torch.ones(1, 4)).sum().backward()
        self.assertIsNotNone(self.layer.captured.grad)

    def test_acceptance_backtracks_package(self):
        old = self.layer.codes.clone()
        proposal = [(g, 0, (int(old.reshape(-1, 2)[g, 0]) + 1) % 4, -1) for g in (0, 1)]
        def measured():
            changes = int((self.layer.codes != old).sum())
            return {0: 1., 1: .5, 2: 2.}[changes]
        result = accept_proposal(self.layer, proposal, measured)
        self.assertEqual(result['accepted'], 1)
        self.assertEqual(len(result['attempts']), 2)

    def test_checkpointed_block_loss_gradients(self):
        from torch.utils.checkpoint import checkpoint
        class Block(nn.Module):
            def __init__(self, layer):
                super().__init__()
                self.linear = layer
            def forward(self, x, c):
                return x + self.linear(x)
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.blocks = nn.ModuleList([Block(AdditiveLinear(torch.randn(1, 2, 4, 1, 2) * .1,
                      torch.randint(4, (4, 2, 2)), torch.ones(4, 1, 1, 1), torch.zeros(4))) for _ in range(3)])
                for block in self.blocks:
                    original = block.forward
                    def wrapped(x, c, original=original):
                        return checkpoint(original, x, c, use_reentrant=False)
                    block.forward = wrapped
            def forward(self, x):
                for block in self.blocks:
                    x = block(x, None)
                return x
        model = Model()
        with block_outputs(model, True) as outputs:
            model(torch.randn(1, 4))
        sum(value.square().sum() for value in outputs).backward()
        self.assertEqual(len(outputs), 3)
        self.assertTrue(all(block.linear.books.grad is not None for block in model.blocks))


if __name__ == '__main__':
    unittest.main()
