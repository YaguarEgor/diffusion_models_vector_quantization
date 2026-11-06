# Quantization of Diffusion Models

Research on compact weight representations for diffusion models: the completed
Stable Diffusion 1.5 coursework and the current DiT-XL/2 diploma project.

## Current Work: Diffusion Transformers

The current direction is adapting vector quantization to DiT, not improving RVQ
at any cost. VQDM was transferred to all 140 matrices in the 28 DiT-XL/2 blocks.
Subsequent experiments examine temporal sensitivity, separate temporal codebooks,
training objectives, discrete assignments, and codebook refinement through the network.

- [Research context](PROJECT_CONTEXT.md)
- [Experiments, protocols and limitations](docs/EXPERIMENTS.md)
- [Processed scientific tables](diploma/run_summaries/README.md)
- [Article draft](diploma/main.pdf) and [LaTeX](diploma/main.tex)
- [Local setup and rebuild instructions](RESTORE.md)

### Talks

| Topic | Slides | Detailed Speaker Notes |
|---|---|---|
| DiT experiments: setup, interventions, results and interpretation | [20-slide PDF](diploma/presentation/dit_results_update/dit_results_update.pdf) | [20-page PDF](diploma/presentation/dit_results_update/speaker_notes.pdf) |
| VQ4DiT and DPQ: architecture, calibration, assignments and codebook storage | [28-slide PDF](diploma/presentation/vq4dit_dpq/review.pdf) | [28-page PDF](diploma/presentation/vq4dit_dpq/speaker_notes.pdf) |

### Results In Brief

The initial transfer reduces logical whole-DiT storage from a 32-bit baseline
to 4.1501 bits per parameter, but worsens FID from 6.192 to 15.726 on a common
10,000-image DDPM50 protocol. Temporal and structural diagnostics motivated
several refinements; the strongest current signal comes from fitting codebooks
to the output of the original network.

A three-block pilot reduced the final paired-generation latent error by 4.3–5.6%.
Fitting all codebooks of the completed VQDM reduced held-out noise-prediction
error by 13.33% without increasing bit width. These are different measurements.
The newer FID of 11.045 uses 5,000 generated images and must not be interpreted
as a measured gain over the older 10,000-image result. A matched initial-model
control remains necessary in the saved scientific record.

### Algorithm Cores

- [Additive codebooks and discrete proposals](diploma/dit_loss_recode/)
- [Output objectives, deployment rounding and trainable alternatives](diploma/dit_objective_followup/)
- [Independent DPQ reconstruction and reference searches](diploma/dpq_reproduction/)
- [VQ4DiT equations and CPU tests](diploma/vq4dit_reproduction/)

Author repositories for DPQ and VQ4DiT are pinned submodules. Author code,
independent reconstructions and inspired experiments are deliberately distinguished.
See [DPQ fidelity boundaries](diploma/dpq_reproduction/fidelity.md).

## Completed Coursework

Stable Diffusion 1.5, activation-aware RVQ, Hadamard transforms and code refinement:
[coursework PDF](coursework/coursework_paper/main.pdf),
[LaTeX](coursework/coursework_paper/main.tex),
[processed results](coursework/run_summaries/README.md).
The coursework archive is separate from the DiT study and uses different protocols.

## Public Release Boundary

This is a scientific publication set, not a complete operational backup.
It includes papers, talks, independent algorithm cores, tests and anonymized
result tables. It does not add cluster-account details, server paths, node names,
scheduler receipts or raw cluster snapshots. Credentials, environments, prompt
datasets, model checkpoints and full ImageNet data are not included.
The full private working snapshot must be transferred separately when changing machines.
