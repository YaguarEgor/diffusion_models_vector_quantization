import copy
import importlib
import logging
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent / 'upstream'))
from paper_pq import PaperPQLinear, compress_pool, decode, fit_groups, nearest


class PaperTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(81)
        torch.set_num_threads(2)

    def test_decode_projection_and_gradient(self):
        pool = torch.randn(5, 2, requires_grad=True)
        projection = torch.tensor([[0, 1, 2], [3, 4, 0]])
        codes = torch.tensor([[0, 2], [1, 0]], dtype=torch.uint8)
        actual = decode(codes, pool, projection)
        expected = torch.stack([torch.cat([pool[0], pool[4]]), torch.cat([pool[2], pool[3]])])
        self.assertTrue(torch.equal(actual, expected))
        actual.square().sum().backward()
        self.assertTrue(torch.isfinite(pool.grad).all())

    def test_nearest_matches_brute_force(self):
        data, centers, x = torch.randn(3, 7, 2), torch.randn(3, 4, 2), torch.randn(9, 3, 2)
        gram = torch.einsum('ngd,nge->gde', x, x) / len(x)
        ids = nearest(data, centers, gram, rows=2)
        delta = data[:, :, None] - centers[:, None]
        distances = torch.einsum('gokd,gde,goke->gok', delta, gram, delta)
        self.assertTrue(torch.equal(ids, distances.argmin(-1)))

    def test_kmeans_fit_reproducible(self):
        data = torch.randn(3, 24, 2)
        rng = torch.get_rng_state()
        a, ids = fit_groups(data, k=4, iterations=3)
        torch.set_rng_state(rng)
        b, ids2 = fit_groups(data, k=4, iterations=3)
        self.assertTrue(torch.equal(a, b))
        self.assertTrue(torch.equal(ids, ids2))
        self.assertTrue(torch.equal(ids, nearest(data, a)))

    def test_pool_selection_projection_capacity(self):
        books = torch.tensor([[[0., 0.], [.01, 0.], [1., 0.]], [[0., 0.], [2., 0.], [3., 0.]]])
        codes = torch.tensor([[0, 0, 0, 1], [0, 0, 1, 2]])
        pool, projection, meta = compress_pool(books, codes, capacity=2, threshold=.05)
        self.assertEqual(len(pool), 2)
        self.assertEqual(projection.shape, (2, 3))
        self.assertEqual(int(projection.max()), 1)
        self.assertEqual(int(projection[0, 0]), int(projection[1, 0]))
        self.assertGreater(meta['projection_fallbacks'], 0)

    def test_e_step_reduces_each_subspace_objective(self):
        base = nn.Linear(8, 16)
        layer, _ = PaperPQLinear.initialize(base, d=2, k=4, iterations=3, threshold=.001)
        x = torch.randn(11, 8)
        def objective():
            delta = (layer.weight() - base.weight).reshape(16, 4, 2)
            outputs = torch.einsum('ogd,ngd->nog', delta, x.reshape(11, 4, 2))
            return outputs.square().sum().item()
        before = objective()
        layer.e_step(x)
        self.assertLessEqual(objective(), before + 1e-5)
        self.assertEqual(layer.e_calls, 1)
        self.assertEqual(layer.codes.dtype, torch.uint8)

    def test_m_step_only_pool(self):
        layer, _ = PaperPQLinear.initialize(nn.Linear(8, 16), d=2, k=4, iterations=2, threshold=.001)
        before = copy.deepcopy(layer.state_dict())
        opt = torch.optim.AdamW([layer.pool], lr=1e-4, weight_decay=0)
        layer(torch.randn(3, 8)).square().sum().backward()
        opt.step()
        self.assertFalse(torch.equal(before['pool'], layer.pool))
        for key in ('codes', 'projection', 'bias', 'reference'):
            if key in before:
                self.assertTrue(torch.equal(before[key], layer.state_dict()[key]))

    def test_dead_reinitialization_and_state_roundtrip(self):
        pool = torch.tensor([[0., 0.], [1., 1.], [2., 2.], [3., 3.]])
        projection = torch.arange(4).repeat(2, 1)
        codes = torch.zeros(2, 8, dtype=torch.uint8)
        layer = PaperPQLinear(pool, projection, codes, torch.randn(8, 4), dead_steps=1, reservoir_size=16)
        self.assertTrue(layer.maintain(refresh=2))
        self.assertEqual(int(layer.age.max()), 0)
        self.assertLess(int(layer.projection.max()), len(pool))
        clone = copy.deepcopy(layer)
        clone.load_state_dict(layer.state_dict())
        self.assertTrue(torch.equal(clone.weight(), layer.weight()))
        self.assertNotIn('reference', layer.state_dict())

    def test_checkpoint_does_not_repeat_e_step(self):
        from checkpointing import attach_checkpointing
        class Block(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc, _ = PaperPQLinear.initialize(nn.Linear(8, 8), d=2, k=4, iterations=2, threshold=.001)
            def forward(self, x, c):
                return self.fc(x).sin() + x
        model = nn.Module()
        model.blocks = nn.ModuleList([Block(), Block()])
        for block in model.blocks:
            block.fc.e_enabled = True
        attach_checkpointing(model, 'paper')
        x = torch.randn(2, 3, 8)
        for block in model.blocks:
            x = block(x, None)
        x.square().sum().backward()
        self.assertTrue(all(block.fc.e_calls == 1 for block in model.blocks))
        self.assertTrue(all(block.fc.pool.grad is not None for block in model.blocks))


class UpstreamAuditTests(unittest.TestCase):
    def test_forces_one_requested_iteration(self):
        module = importlib.import_module('pqf.compressed_layers.CompressedLinear')
        original = module.kmeans
        calls = []
        def observed(*args, **kwargs):
            calls.append(kwargs['n_iters'])
            return original(*args, **kwargs)
        with patch.object(module, 'kmeans', observed):
            layer = module.CompressedLinear.from_uncompressed(nn.Linear(4, 16), 4, 20, observed, 2,
                        logger=logging.getLogger('dpq_test'), type='new')
        self.assertEqual(calls, [1, 1])
        self.assertEqual(tuple(layer.codebook.shape), (2, 4, 2))

    def test_pq_activation_method_is_broken(self):
        module = importlib.import_module('pqf.compressed_layers.CompressedLinear')
        layer = module.CompressedLinear(torch.zeros(2, 8, dtype=torch.int64), torch.randn(2, 4, 2))
        with self.assertRaises((RuntimeError, ValueError)):
            layer.get_loss_act(torch.randn(8, 4), torch.randn(2, 3, 4))

    def test_native_fixed_codes_and_pool_layout(self):
        module = importlib.import_module('pqf.compressed_layers.CompressedLinear')
        codes = torch.randint(4, (2, 8))
        layer = module.CompressedLinear(codes, torch.randn(2, 4, 2), torch.zeros(8))
        original = layer.codes_matrix.clone()
        opt = torch.optim.AdamW(layer.parameters(), lr=1e-4, weight_decay=0)
        layer(torch.randn(3, 4)).square().mean().backward()
        opt.step()
        self.assertTrue(torch.equal(original, layer.codes_matrix))
        self.assertTrue(layer.bias.requires_grad)


if __name__ == '__main__':
    unittest.main()
