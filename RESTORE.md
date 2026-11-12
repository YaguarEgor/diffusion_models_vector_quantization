# Public Checkout: Setup and Rebuilding

This public edition restores the scientific materials and independently testable
algorithm cores. It is not a backup of private infrastructure or trained checkpoints.

## Clone With Pinned Author Code

```bash
git -c core.autocrlf=false clone --recurse-submodules https://github.com/YaguarEgor/diffusion_models_vector_quantization.git
cd diffusion_models_vector_quantization
```

For an existing checkout:

```bash
git -c core.autocrlf=false submodule update --init --recursive
```

Do not use `submodule update --remote`: it changes the author revisions.
The pinned DPQ revision is `12ee544b7dbaf48aecafad5c77c0a6b49be54e25`;
VQ4DiT is `104c62c32c8e09f2eca1b10b3c9223871e43b6a0`.

## Local Environment

Python 3.13 was used for the saved local checks.

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-local.txt
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
```

This is a CPU analysis/test environment, not a prescribed GPU training environment.

```bash
python -B -m unittest discover -s diploma/dit_loss_recode -p "test_*.py"
python -B -m unittest discover -s diploma/dit_objective_followup -p "test_*.py"
python -B -m unittest discover -s diploma/dpq_reproduction -p "test_*.py"
python -B -m unittest discover -s diploma/vq4dit_reproduction -p "test_*.py"
```

## Read the Research Context

Start with [PROJECT_CONTEXT.md](PROJECT_CONTEXT.md),
[docs/EXPERIMENTS.md](docs/EXPERIMENTS.md) and [AGENTS.md](AGENTS.md).
The experiment tables are anonymized: scientific parameters and measurements
remain, while operational identifiers and locations are removed.
Saved experiment status is historical, not live monitoring.

## Rebuild the Presentations

Install XeLaTeX, Arial and Poppler (`pdfinfo`, `pdftotext`, `pdftoppm`).
Each talk includes its LaTeX source, final PDF, matching speaker notes and
the figure/table assets required for a standalone rebuild.

For the experiment talk, work in `diploma/presentation/dit_results_update`:

```powershell
New-Item -ItemType Directory -Force _build
xelatex -interaction=nonstopmode -halt-on-error -output-directory=_build dit_results_update.tex
xelatex -interaction=nonstopmode -halt-on-error -output-directory=_build dit_results_update.tex
xelatex -interaction=nonstopmode -halt-on-error -output-directory=_build speaker_notes.tex
xelatex -interaction=nonstopmode -halt-on-error -output-directory=_build speaker_notes.tex
python -B verify_pdf.py
```

For the article review, work in `diploma/presentation/vq4dit_dpq` and use
`review.tex` instead of `dit_results_update.tex`; compile the notes twice
and run its `verify_pdf.py` in the same way.

The article and practicum-specific slides are maintained in
[my_first_article](https://github.com/YaguarEgor/my_first_article). Its README describes rebuilding
the current manuscript and talks. The earlier draft and its exact dependencies
are preserved there under `archive/pre_kd_draft/`. Research talks above remain here.

## What Must Be Restored Privately

Trained models, ImageNet, large activation/image caches, private launch recipes,
infrastructure settings and complete machine-migration records are not on GitHub.
Transfer them independently or recover access through the relevant administrator.
Never attach passwords, private keys or access tokens to a public issue or commit.
The public checkout alone does not reproduce access to an existing remote environment.

