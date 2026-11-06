import copy
import unittest
from unittest.mock import patch

import torch
from torch import nn

from paper_pq import PaperPQLinear, nearest, sample_rows
from search_reference import e_step_reference, maintenance_reference, nearest_reference


class FastSearchTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(713)
        torch.set_num_threads(2)

    def test_bounded_tiles_match_reference(self):
        data, books = torch.randn(7, 19, 4), torch.randn(7, 11, 4)
        x = torch.randn(23, 7, 4)
        gram = torch.einsum('ngd,nge->gde', x, x)
        for metric in (None, gram, torch.zeros_like(gram)):
            expected = nearest_reference(data, books, metric, rows=3)
            for budget in (44, 88, 1024, 65536):
                actual = nearest(data, books, metric, rows=8, group_chunk=3,
                                 max_distance_bytes=budget)
                self.assertTrue(torch.equal(actual, expected))

    def test_first_tie_and_half_inputs(self):
        books = torch.ones(2, 5, 2, dtype=torch.float16)
        self.assertTrue(torch.equal(nearest(torch.zeros(2, 9, 2), books),
                                    torch.zeros(2, 9, dtype=torch.long)))

    def test_invalid_shapes_and_budget(self):
        data, books = torch.zeros(2, 4, 3), torch.zeros(2, 8, 3)
        for kwargs in ({'rows': 0}, {'group_chunk': 0}, {'max_distance_bytes': 31},
                       {'gram': torch.zeros(3, 3, 3)}):
            with self.assertRaises(ValueError):
                nearest(data, books, **kwargs)
        with self.assertRaises(ValueError):
            nearest(data[:, :0], books)

    def test_e_step_indices_counts_and_gradients(self):
        layer, _ = PaperPQLinear.initialize(nn.Linear(32, 40), d=4, k=8, iterations=2)
        old = copy.deepcopy(layer)
        x = torch.randn(3, 9, 32)
        expected = e_step_reference(old, x)
        actual = layer.e_step(x)
        self.assertEqual(actual.item(), expected)
        self.assertTrue(torch.equal(layer.codes, old.codes))
        self.assertTrue(torch.equal(layer.weight(), old.weight()))
        layer(x).square().mean().backward()
        old(x).square().mean().backward()
        self.assertTrue(torch.equal(layer.pool.grad, old.pool.grad))
        self.assertNotIn('last_changed', layer.state_dict())

    def test_no_host_scalar_reads_in_e_step(self):
        layer, _ = PaperPQLinear.initialize(nn.Linear(16, 32), d=4, k=4, iterations=1)
        x = torch.randn(4, 16)
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as trace:
            layer.e_step(x)
        names = {event.key for event in trace.key_averages()}
        self.assertNotIn('aten::item', names)
        self.assertNotIn('aten::_local_scalar_dense', names)

    def test_maintenance_ages_match_without_full_permutation(self):
        layer, _ = PaperPQLinear.initialize(nn.Linear(16, 32), d=4, k=4, iterations=1)
        old = copy.deepcopy(layer)
        layer.age.fill_(13)
        old.age.copy_(layer.age)
        maintenance_reference(old, refresh=0)
        with patch('torch.randperm', side_effect=AssertionError('Full permutation')):
            self.assertFalse(layer.maintain(refresh=5, sample_seed=521))
        self.assertTrue(torch.equal(layer.age, old.age))
        clone = copy.deepcopy(layer)
        layer.maintain(sample_seed=733)
        clone.maintain(sample_seed=733)
        self.assertTrue(torch.equal(layer.reservoir, clone.reservoir))

    def test_sample_unique_seeded_and_bounded(self):
        ids = sample_rows(100000000, 256, 'cpu', 177)
        self.assertEqual(len(ids.unique()), 256)
        self.assertTrue(torch.equal(ids, sample_rows(100000000, 256, 'cpu', 177)))
        self.assertFalse(torch.equal(ids, sample_rows(100000000, 256, 'cpu', 178)))
        self.assertGreaterEqual(int(ids.min()), 0)
        self.assertLess(int(ids.max()), 100000000)
        self.assertEqual(sample_rows(0, 0, 'cpu', 1).numel(), 0)
        with self.assertRaises(ValueError):
            sample_rows(3, 4, 'cpu', 1)


if __name__ == '__main__':
    unittest.main()
