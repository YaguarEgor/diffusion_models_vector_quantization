"""Single-device VQDM Lloyd updates without CUDA atomic reductions."""

import torch
from torch.nn import functional as F


@torch.no_grad()
def cluster_means(data, indices, previous, chunk=8192):
    if (data.ndim != 2 or previous.ndim != 2 or data.shape[1] != previous.shape[1]
            or indices.shape != (len(data),) or chunk < 1):
        raise ValueError('Invalid cluster reduction shapes')
    k = len(previous)
    sums = torch.zeros_like(previous)
    counts = torch.zeros(k, device=data.device, dtype=data.dtype)
    # Fixed-order GEMMs replace index_reduce_ / floating-point atomic additions.
    for start in range(0, len(data), chunk):
        membership = F.one_hot(indices[start:start + chunk], k).to(data.dtype)
        sums.add_(membership.T @ data[start:start + chunk])
        counts.add_(membership.sum(0))
    means = sums / counts.clamp_min(1)[:, None]
    return torch.where(counts[:, None] > 0, means, previous)


@torch.no_grad()
def fit_kmeans(data, k, max_iter=1000, check_every=10, rtol=1e-6, atol=1e-8,
               greedy_init=False, block_size_vals=2**24, devices=None):
    if (data.ndim != 2 or not 0 < k <= len(data) or max_iter < 1
            or check_every < 1 or block_size_vals < k):
        raise ValueError('Invalid K-means arguments')
    if devices is not None and (len(devices) != 1 or torch.device(devices[0]) != data.device):
        raise ValueError('This deterministic initializer supports one source device')
    if not torch.isfinite(data).all():
        raise ValueError('K-means input is nonfinite')
    if greedy_init:
        clusters = torch.zeros(k, data.shape[1], device=data.device, dtype=data.dtype)
        distances = torch.full((len(data),), torch.inf, device=data.device, dtype=data.dtype)
        norm = data.norm(dim=1).square()
        for index in range(k):
            clusters[index] = data[distances.argmax()]
            distances = torch.minimum(distances, norm - 2 * data @ clusters[index]
                                       + clusters[index].norm().square())
    else:
        # Native VQDM draws this permutation on CPU; retain its RNG convention.
        clusters = data[torch.randperm(len(data))[:k]].clone()
    block = block_size_vals // k

    def assign(centers):
        ids = torch.empty(len(data), dtype=torch.long, device=data.device)
        norm = torch.bmm(centers[:, None], centers[:, :, None]).flatten()
        for start in range(0, len(data), block):
            ids[start:start + block] = torch.addmm(norm, data[start:start + block],
                                                  centers.T, beta=-.5).argmax(1)
        return ids

    for iteration in range(max_iter):
        ids = assign(clusters)
        updated = cluster_means(data, ids, clusters)
        if iteration % check_every == 0 and torch.allclose(updated, clusters, rtol=rtol, atol=atol):
            break
        clusters = updated
    ids = assign(clusters)
    return clusters, ids, clusters[ids]
