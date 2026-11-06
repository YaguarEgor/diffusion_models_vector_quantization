"""Independent reconstruction of DPQ sections 4.2/4.3, not author-released code."""

import random

import numpy as np
from scipy.spatial import cKDTree
import torch
from torch import nn
from torch.nn import functional as F

IMPLEMENTATION = 'bounded-search-v2-stateless-reservoir'


def sample_rows(population, count, device, seed):
    """Uniform sampling without replacement, without a population-size permutation."""
    if not 0 <= count <= population:
        raise ValueError('Invalid sample size')
    return torch.tensor(random.Random(seed).sample(range(population), count),
                        dtype=torch.long, device=device)


def decode(codes, pool, projection):
    groups, outputs = codes.shape
    ids = projection.gather(1, codes.long()).t().reshape(-1).long()
    return F.embedding(ids, pool).reshape(outputs, groups * pool.shape[1])


@torch.no_grad()
def nearest(data, centers, gram=None, rows=1024, group_chunk=64,
            max_distance_bytes=64 * 1024**2):
    """Independent groups, using Euclidean or exact within-group XTX distances."""
    if data.ndim != 3 or centers.ndim != 3 or data.shape[0] != centers.shape[0]:
        raise ValueError('Expected data [G,O,d] and codebooks [G,K,d]')
    if data.shape[-1] != centers.shape[-1]:
        raise ValueError('Subvector dimensions differ')
    if min(*data.shape, centers.shape[1], rows, group_chunk) <= 0:
        raise ValueError('Search dimensions and chunks must be positive')
    if gram is not None and gram.shape != (data.shape[0], data.shape[-1], data.shape[-1]):
        raise ValueError('Invalid within-group covariance')
    candidates = centers.shape[1]
    elements = max_distance_bytes // 4
    if elements < candidates:
        raise ValueError('Distance budget cannot fit even one candidate row')
    # The largest G-by-O-by-K temporary is bounded, including large reinit pools.
    groups = min(group_chunk, max(1, elements // (min(rows, data.shape[1]) * candidates)))
    result = torch.empty(data.shape[:2], device=data.device, dtype=torch.long)
    for start in range(0, data.shape[0], groups):
        stop = min(start + groups, data.shape[0])
        books = centers[start:stop].float()
        metric = None if gram is None else gram[start:stop].float()
        transformed_books = books if metric is None else torch.bmm(books, metric)
        norms = (transformed_books * books).sum(-1).unsqueeze(1)
        row_chunk = min(rows, elements // ((stop - start) * candidates))
        for row in range(0, data.shape[1], row_chunk):
            vectors = data[start:stop, row:row + row_chunk].float()
            transformed = vectors if metric is None else torch.bmm(vectors, metric)
            distance = torch.bmm(transformed, books.transpose(1, 2))
            distance.mul_(-2).add_((transformed * vectors).sum(-1, keepdim=True)).add_(norms)
            result[start:stop, row:row + row_chunk] = distance.argmin(-1)
    return result


@torch.no_grad()
def fit_groups(data, k=256, iterations=20, group_chunk=8):
    """Batched Lloyd K-means, avoiding a Python/CUDA loop over each centroid."""
    if data.ndim != 3 or not 1 <= k <= data.shape[1] or iterations < 1:
        raise ValueError('Invalid grouped K-means dimensions or iteration count')
    books, assignments = [], []
    for start in range(0, len(data), group_chunk):
        values = data[start:start + group_chunk].float()
        initial = torch.stack([torch.randperm(values.shape[1], device=values.device)[:k]
                               for _ in range(len(values))])
        centers = values.gather(1, initial[:, :, None].expand(-1, -1, values.shape[2])).clone()
        for _ in range(iterations):
            codes = nearest(values, centers)
            # Dense memberships are small because groups and output rows are chunked.
            sums = torch.zeros_like(centers)
            counts = torch.zeros(len(values), k, device=values.device)
            for row in range(0, values.shape[1], 256):
                membership = F.one_hot(codes[:, row:row + 256], k).float()
                sums += torch.bmm(membership.transpose(1, 2), values[:, row:row + 256])
                counts += membership.sum(1)
            # Empty centers retain their old value. Algorithm 1 handles aging later.
            centers = torch.where(counts[:, :, None] > 0,
                                  sums / counts.clamp_min(1)[:, :, None], centers)
        books.append(centers)
        assignments.append(nearest(values, centers))
    return torch.cat(books), torch.cat(assignments)


def compress_pool(books, codes, capacity, threshold=.05):
    """Frequency-ordered greedy pool; exact radius checks with periodic KD trees.

    Pool centers are selected, not averaged. Projection chooses the closest usage
    frequency within the radius, then distance and index; a full-pool/no-neighbor
    case falls back to nearest distance. These tie/fallback rules are explicit
    reconstruction choices because the paper does not uniquely specify them.
    """
    values = books.detach().float().cpu().reshape(-1, books.shape[-1]).numpy().astype(np.float64)
    groups, k, d = books.shape
    frequencies = np.concatenate([np.bincount(row, minlength=k)
                                  for row in codes.detach().cpu().numpy()])
    order = np.argsort(-frequencies, kind='stable')
    if threshold <= 0 or not 1 <= capacity <= min(len(values), 65536):
        raise ValueError('Invalid pool capacity/threshold')
    selected, tree, committed = [], None, 0
    for index in order:
        if len(selected) == capacity:
            break
        if tree is not None and tree.query(values[index], k=1)[0] < threshold:
            continue
        recent = values[selected[committed:]]
        if len(recent) and np.linalg.norm(recent - values[index], axis=1).min() < threshold:
            continue
        selected.append(int(index))
        if len(selected) - committed >= 128:
            tree = cKDTree(values[selected])
            committed = len(selected)
    pool = values[selected]
    pool_frequency = frequencies[selected]
    tree = cKDTree(pool)
    projection = np.empty(len(values), dtype=np.int64)
    fallbacks = 0
    for i, value in enumerate(values):
        neighbors = tree.query_ball_point(value, threshold)
        # query_ball_point includes equality; Eq.(4)/(5) use a strict threshold.
        neighbors = [j for j in neighbors if np.linalg.norm(pool[j] - value) < threshold]
        if not neighbors:
            neighbors = [int(tree.query(value)[1])]
            fallbacks += 1
        projection[i] = min(neighbors, key=lambda j: (abs(int(pool_frequency[j]) - int(frequencies[i])),
                                                    float(np.square(pool[j] - value).sum()), j))
    return (torch.tensor(pool, dtype=torch.float32, device=books.device),
            torch.tensor(projection.reshape(groups, k), dtype=torch.int32, device=books.device),
            dict(input_centroids=len(values), pool_centroids=len(pool), capacity=capacity,
                 threshold=threshold, projection_fallbacks=fallbacks))


class PaperPQLinear(nn.Module):
    def __init__(self, pool, projection, codes, reference, bias=None, dead_steps=256,
                 reservoir_size=None):
        super().__init__()
        if projection.shape[0] != codes.shape[0] or len(pool) > 65536:
            raise ValueError('Invalid PQ layout or INT16 projection capacity')
        self.pool = nn.Parameter(pool.detach().float().clone())
        self.bias = None if bias is None else nn.Parameter(bias.detach().clone(), requires_grad=False)
        self.register_buffer('projection', projection.detach().to(torch.int32).clone())
        self.register_buffer('codes', codes.detach().to(torch.uint8).clone())
        self.register_buffer('age', torch.zeros(len(pool), dtype=torch.int64, device=pool.device))
        self.register_buffer('reference', reference.detach().float().clone(), persistent=False)
        d = pool.shape[1]
        vectors = reference.detach().reshape(-1, d)
        size = min(len(vectors), reservoir_size or max(4096, 2 * len(pool)))
        ids = sample_rows(len(vectors), size, pool.device, torch.initial_seed())
        self.register_buffer('reservoir', vectors[ids].float().clone())
        self.dead_steps = dead_steps
        self.e_enabled = False
        self.e_calls = 0
        self.register_buffer('last_changed', torch.zeros((), dtype=torch.long, device=pool.device),
                             persistent=False)

    @classmethod
    def initialize(cls, layer, d=4, k=256, iterations=20, threshold=.05):
        weight = layer.weight.detach().float()
        o, inputs = weight.shape
        if inputs % d or not 1 <= k <= min(256, o):
            raise ValueError('The profile does not divide input width, or too few rows for K')
        values = weight.reshape(o, inputs // d, d).transpose(0, 1)
        books, codes = fit_groups(values, k, iterations)
        capacity = max(1, min(65536, books.shape[0] * k, weight.numel() // (16 * d * d)))
        pool, projection, metadata = compress_pool(books, codes, capacity, threshold)
        result = cls(pool, projection, codes, weight, layer.bias)
        return result, metadata

    def weight(self):
        return decode(self.codes, self.pool, self.projection)

    @torch.no_grad()
    def e_step(self, x):
        d = self.pool.shape[-1]
        inputs = x.detach().float().reshape(-1, self.codes.shape[0], d)
        # Only d-by-d XTX blocks are needed for the paper's per-subspace Eq.(6).
        gram = torch.einsum('ngd,nge->gde', inputs, inputs) / max(1, len(inputs))
        reference = self.reference.reshape(self.reference.shape[0], -1, d).transpose(0, 1)
        candidates = self.pool.detach()[self.projection.long()]
        codes = nearest(reference, candidates, gram)
        # Keep the diagnostic on device: reading it on the host would sync CUDA.
        changed = (codes != self.codes).sum()
        self.last_changed.copy_(changed)
        self.codes.copy_(codes.to(torch.uint8))
        self.e_calls += 1
        return changed

    def forward(self, x):
        if self.e_enabled and torch.is_grad_enabled():
            self.e_step(x)
        return F.linear(x, self.weight(), self.bias)

    @torch.no_grad()
    def maintain(self, refresh=256, trigger=.03, sample_seed=None):
        d = self.pool.shape[1]
        source = self.reference.reshape(-1, d)
        count = min(refresh, len(self.reservoir), len(source))
        if count:
            seed = random.randrange(2**63) if sample_seed is None else sample_seed
            ids = sample_rows(len(source), count, source.device, seed)
            self.reservoir.copy_(torch.cat((self.reservoir[count:], source[ids])))
        ids = self.projection.gather(1, self.codes.long()).reshape(-1).long()
        used = torch.bincount(ids, minlength=len(self.pool)) > 0
        self.age.add_(1)
        self.age.masked_fill_(used, 0)
        dead = self.age >= self.dead_steps
        if float(dead.float().mean()) <= trigger:
            return False
        candidates = torch.cat((self.reservoir, self.pool.detach()[~dead]))
        if len(candidates) < len(self.pool):
            raise ValueError('Reservoir cannot support reinitialization')
        # Algorithm 1 specifies k-means++; use its seeded sequential seeding.
        from sklearn.cluster import kmeans_plusplus
        seed = int(torch.randint(0, 2**31 - 1, ()).item())
        values, _ = kmeans_plusplus(candidates.cpu().numpy(), n_clusters=len(self.pool), random_state=seed)
        old_books = self.pool.detach()[self.projection.long()]
        new_pool = torch.as_tensor(values, device=self.pool.device, dtype=self.pool.dtype)
        # Keep local codewords/index semantics via a fresh projection after the
        # pool changes; a later E-step can change the weight assignments too.
        new_projection = nearest(old_books.reshape(1, -1, d), new_pool[None])[0]
        self.pool.copy_(new_pool)
        self.projection.copy_(new_projection.reshape_as(self.projection).int())
        self.age.zero_()
        return True

    def storage_bits(self):
        return dict(indices=self.codes.numel() * 8, projection=self.projection.numel() * 16,
                    codebooks=self.pool.numel() * 16,
                    bias=0 if self.bias is None else self.bias.numel() * 16)
