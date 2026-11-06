"""Pre-optimization kernels, retained only as independent benchmark oracles."""

import torch


@torch.no_grad()
def nearest_reference(data, centers, gram=None, rows=256):
    if gram is None:
        gram = torch.eye(data.shape[-1], device=data.device).expand(data.shape[0], -1, -1)
    result = []
    for start in range(0, data.shape[1], rows):
        vectors = data[:, start:start + rows].float()
        transformed = torch.bmm(vectors, gram.float())
        transformed_centers = torch.bmm(centers.float(), gram.float())
        distance = ((transformed * vectors).sum(-1, keepdim=True)
                    - 2 * torch.bmm(transformed, centers.float().transpose(1, 2))
                    + (transformed_centers * centers.float()).sum(-1).unsqueeze(1))
        result.append(distance.argmin(-1))
    return torch.cat(result, dim=1)


@torch.no_grad()
def e_step_reference(layer, x):
    d = layer.pool.shape[-1]
    inputs = x.detach().float().reshape(-1, layer.codes.shape[0], d)
    gram = torch.einsum('ngd,nge->gde', inputs, inputs) / max(1, len(inputs))
    reference = layer.reference.reshape(layer.reference.shape[0], -1, d).transpose(0, 1)
    changed = 0
    for start in range(0, len(reference), 8):
        candidates = layer.pool.detach()[layer.projection[start:start + 8].long()]
        codes = nearest_reference(reference[start:start + 8], candidates, gram[start:start + 8])
        changed += int((codes != layer.codes[start:start + 8]).sum())
        layer.codes[start:start + 8].copy_(codes.to(torch.uint8))
    layer.e_calls += 1
    layer.last_changed.fill_(changed)
    return changed


@torch.no_grad()
def maintenance_reference(layer, refresh=256):
    """Old common path only; refuse to silently skip a triggered reinitialization."""
    source = layer.reference.reshape(-1, layer.pool.shape[1])
    count = min(refresh, len(layer.reservoir), len(source))
    if count:
        ids = torch.randperm(len(source), device=source.device)[:count]
        layer.reservoir.copy_(torch.cat((layer.reservoir[count:], source[ids])))
    used = layer.projection.gather(1, layer.codes.long()).unique().long()
    layer.age.add_(1)
    layer.age[used] = 0
    if float((layer.age >= layer.dead_steps).float().mean()) > .03:
        raise RuntimeError('Benchmark must not time only part of a reinitialization')
    return False
