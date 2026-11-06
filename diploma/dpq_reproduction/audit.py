"""Machine-readable upstream audit. No imports execute the upstream training code."""

import ast
import hashlib
import json
from pathlib import Path
import subprocess

COMMIT = '12ee544b7dbaf48aecafad5c77c0a6b49be54e25'
URL = 'https://github.com/Jie-Shao-NJU/DPQ'
HERE = Path(__file__).resolve().parent


def hash_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024**2), b''):
            h.update(chunk)
    return h.hexdigest()


def git_inventory(repo):
    head = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=repo, capture_output=True,
                          text=True, check=True).stdout.strip()
    dirty = subprocess.run(['git', 'status', '--porcelain', '--untracked-files=no'], cwd=repo,
                           capture_output=True, text=True, check=True).stdout
    if head != COMMIT or dirty:
        raise ValueError('The audited author source must be pinned and unmodified')
    tracked = subprocess.run(['git', 'ls-files', '-z'], cwd=repo, capture_output=True,
                             text=True, check=True).stdout.split('\0')
    return {name: hash_file(repo / name) for name in tracked if name}


def verify_seal(repo, manifest):
    report = json.loads(Path(manifest).read_text())
    if report.get('commit') != COMMIT or report.get('upstream') != URL:
        raise ValueError('Unexpected sealed upstream revision')
    hashes = report.get('tracked_sha256', {})
    if not hashes or not any(name.endswith('.py') for name in hashes):
        raise ValueError('Empty upstream source manifest')
    for name, checksum in hashes.items():
        path = (repo / name).resolve()
        path.relative_to(repo.resolve())
        if not path.is_file() or hash_file(path) != checksum:
            raise ValueError(f'Sealed upstream file changed: {name}')
    actual = {p.relative_to(repo).as_posix() for p in repo.rglob('*.py')
              if '.git' not in p.relative_to(repo).parts}
    if actual != {name for name in hashes if name.endswith('.py')}:
        raise ValueError('Upstream Python source inventory changed')
    return hashes


def seal(repo=HERE / 'upstream', manifest=None):
    repo = Path(repo)
    manifest = Path(manifest) if manifest is not None else repo.parent / 'upstream-manifest.json'
    hashes = git_inventory(repo)
    if manifest.exists():
        if verify_seal(repo, manifest) != hashes:
            raise ValueError('Existing upstream seal differs from git inventory')
    else:
        with manifest.open('x', encoding='utf-8') as handle:
            json.dump(dict(upstream=URL, commit=COMMIT, tracked_sha256=hashes), handle, indent=2)
            handle.write('\n')
    return manifest


def inspect(repo=HERE / 'upstream', manifest=None):
    repo = Path(repo)
    manifest = Path(manifest) if manifest is not None else repo.parent / 'upstream-manifest.json'
    try:
        tracked = git_inventory(repo)
    except FileNotFoundError:
        # Compute nodes need no git binary; the login-node seal is still verified.
        tracked = verify_seal(repo, manifest)
    else:
        if manifest.exists() and verify_seal(repo, manifest) != tracked:
            raise ValueError('Upstream seal differs from current git inventory')
    def evidence(filename, substring):
        return dict(file=filename, lines=[i for i, line in enumerate((repo / filename).read_text().splitlines(), 1)
                                         if substring in line])
    issues = [
        dict(id='forced_one_iteration', severity='algorithm',
             detail='PQ branch passes n_iters=1 instead of requested k_means_n_iters.',
             evidence=evidence('pqf/compressed_layers/CompressedLinear.py', 'n_iters=1')),
        dict(id='no_e_step_in_training', severity='algorithm',
             detail='Published train() calls DDPM backward/AdamW, with no assignment update.',
             evidence=evidence('pq/utils_traineval.py', 'def train(')),
        dict(id='broken_activation_assignment', severity='algorithm',
             detail='get_loss_act assumes a 2D shared codebook and fixed d=4; not compatible with PQ books [G,K,d].',
             evidence=evidence('pqf/compressed_layers/CompressedLinear.py', 'def get_loss_act')),
        dict(id='broken_block_calibration_branch', severity='algorithm',
             detail='gptq branch uses nonexistent attn.fc1/fc2 and incorrect residual/input handling.',
             evidence=evidence('models.py', 'self.attn.fc')),
        dict(id='codebook_pool_not_found', severity='missing',
             detail='No section4.2 centroid-pool/projection compression found in tracked Python sources.'),
        dict(id='reinitialization_not_found', severity='missing',
             detail='No Algorithm1 reservoir/code-age/dead-centroid mechanism found in tracked Python sources.'),
        dict(id='all_parameters_optimized', severity='protocol',
             detail='Optimizer receives all model.parameters(), including unquantized modules and biases.',
             evidence=evidence('pq/utils_traineval.py', 'AdamW(model.parameters()')),
        dict(id='respaced_training_diffusion', severity='protocol',
             detail='init_model uses sampling step count for training diffusion too; default is 250, not 1000.',
             evidence=evidence('pq/utils_model.py', 'diffusion = create_diffusion')),
        dict(id='incorrect_pq_size_estimate', severity='reporting',
             detail='Size estimator uses codebook.size(0)=number of subspaces, not K, to count index bits.',
             evidence=evidence('pqf/utils/model_size.py', 'math.log(module.size(0)')),
        dict(id='readme_cli_mismatch', severity='launch',
             detail='README --bits is absent from the real parsers; train.py is the original dense DiT trainer.',
             evidence=evidence('pq/utils_model.py', 'def parse_option')),
    ]
    hashes = {name: checksum for name, checksum in tracked.items() if name.endswith('.py')}
    return dict(upstream=URL, commit=COMMIT, status='incomplete_for_paper_reproduction', issues=issues,
                python_source_sha256=hashes,
                note='Absence findings are a source audit, not a claim that authors have no private implementation.')


if __name__ == '__main__':
    print(json.dumps(inspect(), indent=2))
