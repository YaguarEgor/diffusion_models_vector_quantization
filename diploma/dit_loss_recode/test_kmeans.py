import copy
import unittest
from unittest.mock import patch

import torch

from deterministic_kmeans import cluster_means, fit_kmeans


def cuda_smoke():
    from src.aq import QuantizedWeight
    from core import AdditiveLinear
    from torch import nn
    if not torch.are_deterministic_algorithms_enabled():
        raise ValueError('Strict deterministic algorithms must remain enabled')
    reference = nn.Linear(32, 64, device='cuda')
    copies = []
    for _ in range(2):
        torch.manual_seed(1701)
        with patch('src.aq.fit_kmeans', fit_kmeans):
            q = QuantizedWeight(XTX=None, reference_weight=reference.weight, in_group_size=8,
                out_group_size=1, num_codebooks=4, nbits_per_codebook=4,
                input_channel_scales=False, codebook_value_nbits=16, scale_nbits=0,
                max_iter=3, devices=[torch.device('cuda:0')]).eval().requires_grad_(False).half()
        copies.append(copy.deepcopy(q.state_dict()))
    if any(not torch.equal(copies[0][key], copies[1][key]) for key in copies[0]):
        raise ValueError('Repeated CUDA initialization is not bitwise reproducible')
    layer = AdditiveLinear(q.codebooks, q.get_codes(), q.get_scales(), reference.bias)
    if not torch.equal(layer.weight(), q().float()):
        raise ValueError('CUDA reconstruction differs from native stored FP16 weights')
    layer(torch.randn(2, 32, device='cuda')).square().mean().backward()
    if layer.books.grad is None or not torch.isfinite(layer.books.grad).all():
        raise ValueError('CUDA codebook backward is invalid')
    return dict(status='complete', strict_deterministic=True,
                repeated_initialization='bitwise equal', native_fp16_reconstruction='equal',
                codebook_backward='finite', torch=torch.__version__)


class KmeansTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(12)
        torch.set_num_threads(2)

    def test_reduction_and_empty_clusters(self):
        data = torch.randn(17, 3, dtype=torch.float64)
        previous = torch.randn(5, 3, dtype=torch.float64)
        indices = torch.arange(17) % 3
        expected = previous.clone()
        for i in range(3):
            expected[i] = data[indices == i].mean(0)
        for chunk in (1, 7, 64):
            actual = cluster_means(data, indices, previous, chunk=chunk)
            torch.testing.assert_close(actual, expected)
            self.assertTrue(torch.equal(actual[3:], previous[3:]))

    def test_repeated_fit_and_assignments(self):
        data = torch.randn(83, 4)
        for greedy in (False, True):
            torch.manual_seed(53)
            a, ids, restored = fit_kmeans(data, 7, max_iter=5, greedy_init=greedy, block_size_vals=77)
            torch.manual_seed(53)
            b, ids2, _ = fit_kmeans(data, 7, max_iter=5, greedy_init=greedy, block_size_vals=77)
            self.assertTrue(torch.equal(a, b))
            self.assertTrue(torch.equal(ids, ids2))
            self.assertTrue(torch.equal(restored, a[ids]))
            brute = (data[:, None] - a[None]).square().sum(-1).argmin(-1)
            self.assertTrue(torch.equal(ids, brute))

    def test_single_lloyd_update_matches_reference(self):
        data = torch.randn(43, 4, dtype=torch.float64)
        torch.manual_seed(71)
        initial = data[torch.randperm(len(data))[:5]]
        ids = (data[:, None] - initial[None]).square().sum(-1).argmin(-1)
        expected = initial.clone().index_reduce_(0, ids, data, 'mean', include_self=False)
        torch.manual_seed(71)
        actual, _, _ = fit_kmeans(data, 5, max_iter=1)
        torch.testing.assert_close(actual, expected)

    def test_invalid_input(self):
        for k in (0, 4):
            with self.assertRaises(ValueError):
                fit_kmeans(torch.ones(3, 2), k)
        with self.assertRaises(ValueError):
            fit_kmeans(torch.tensor([[float('nan')]]), 1)


if __name__ == '__main__':
    unittest.main()
