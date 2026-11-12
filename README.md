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
- [Article, practicum slides and three-minute talk](https://github.com/YaguarEgor/my_first_article)
- [Local setup and rebuild instructions](RESTORE.md)

### Talks

| Topic | Slides | Detailed Speaker Notes |
|---|---|---|
| DiT experiments: setup, interventions, results and interpretation | [20-slide PDF](diploma/presentation/dit_results_update/dit_results_update.pdf) | [20-page PDF](diploma/presentation/dit_results_update/speaker_notes.pdf) |
| VQ4DiT and DPQ: architecture, calibration, assignments and codebook storage | [28-slide PDF](diploma/presentation/vq4dit_dpq/review.pdf) | [28-page PDF](diploma/presentation/vq4dit_dpq/speaker_notes.pdf) |

### Results In Brief

The completed matched10k experiment uses DiT-XL/2, ImageNet256, DDPM50,
CFG1.5, MSE VAE and the same paired noise/class records (selection seed20261103).
FID is6.097989 for FP,15.401847 for transferred layerwise-only VQDM and
7.814155 after500 codebook-only KD updates. Logical whole-DiT storage stays
4.150078 bits/parameter; assignments, scales and all other parameters are fixed.
Recall falls from0.7115 to0.6770. This is one fixed run, not an across-seed guarantee.
See the [scientific measurement snapshot](diploma/run_summaries/vqdm_kd500_matched_10k_20261112/).

Earlier temporal/structural diagnostics and negative results remain in this
repository. Their older10k or5k cohorts are separate protocols and must not
be mixed with this matched comparison. VQDM already proposes global
teacher-output fine-tuning for U-Net; the present experiment checks a
restricted DiT transfer, not a new distillation algorithm.

The article and practicum-specific slides have moved to
[my_first_article](https://github.com/YaguarEgor/my_first_article), including
historical drafts with operational JSON metadata removed. Research code and scientific data remain here.

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

