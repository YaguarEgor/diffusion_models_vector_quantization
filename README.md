# Quantization of Diffusion Models

This repository contains coursework materials, code and compact experiment summaries for a study of **post-training quantization of Stable Diffusion 1.5**.

Full coursework PDF: [coursework/coursework_paper/main.pdf](coursework/coursework_paper/main.pdf)

LaTeX source: [coursework/coursework_paper/main.tex](coursework/coursework_paper/main.tex)

Experiment summary index: [coursework/run_summaries/README.md](coursework/run_summaries/README.md)

---

## Repository Overview

```text
cursovaya/
|
|-- coursework/
|   |-- coursework_paper/  LaTeX source, logo and final coursework PDF
|   |-- presentation/      research presentation sources and PDFs
|   |-- rvq/               main RVQ implementation and Hadamard sweep
|   |-- vqdm/              VQDM baseline adapted to Stable Diffusion 1.5
|   |-- baselines/         scalar and vector baseline implementations
|   |-- quant/             larger RVQ ablation experiment
|   |-- run_summaries/     compact processed result tables
|   |-- diffusion_quantization_research_map.md
|   `-- requirements.txt   shared Python dependencies
`-- diploma/               reserved for subsequent diploma work
```

---

## Research Goal

The work studies whether residual vector quantization can approach the quality of stronger AVQ/VQDM-style vector quantization while being cheaper to run.

The experiments focus on quantizing the U-Net middle block of Stable Diffusion 1.5. Baselines include scalar PTQ-style methods, lightweight diffusion quantization methods, vanilla vector quantization, AQUATIC-Diff-like quantization and VQDM. The proposed line is an RVQ method with activation-aware optimization, stage-wise codebook sizes, optional Hadamard transforms and post-RVQ code refinement.

---

## Implemented Methods

Scalar baselines in [coursework/baselines/](coursework/baselines/):

* Scalar PTQ;
* Q-Diffusion-lite;
* PTQ4DM-lite;
* BitsFusion-lite.

Vector baselines:

* Vanilla VQ;
* AQUATIC-Diff-lite;
* VQDM.

Proposed method:

* activation-aware RVQ;
* fixed and stage-wise codebook profiles;
* Hadamard/no-Hadamard comparisons;
* full-XTX post-RVQ recoding;
* entropy, utilization and effective-bpp statistics.

---

## Results

Processed results are stored in [coursework/run_summaries/](coursework/run_summaries/). The most important folders are:

* [baselines_20260523_full_midblock](coursework/run_summaries/baselines_20260523_full_midblock): full-midblock scalar baselines;
* [baselines_20260524_cluster_completed](coursework/run_summaries/baselines_20260524_cluster_completed): completed 18-layer baseline runs;
* [baselines_20260525_completed](coursework/run_summaries/baselines_20260525_completed): completed 16-layer vector baselines;
* [rvq_final_20260531](coursework/run_summaries/rvq_final_20260531): final RVQ sweep;
* [rvq_hadamard_clean_20260602](coursework/run_summaries/rvq_hadamard_clean_20260602): isolated Hadamard-block experiment;
* [rvq_jobb_ablation_20260513_14](coursework/run_summaries/rvq_jobb_ablation_20260513_14): earlier RVQ ablations.

The main quality metrics are CLIP image-image similarity, LPIPS, CLIP text-image similarity, PSNR, SSIM and quantization time. Several summaries also report the combined score:

```text
score = CLIP image-image - LPIPS
```

---

## Main Outcome

VQDM remains the strongest baseline in the saved experiments, but the final RVQ variants get much closer to it than the simpler scalar and lightweight vector baselines while avoiding the expensive AVQ-style search. The most useful RVQ modifications were stage-wise codebook sizes, activation-aware training and full-XTX post-RVQ recoding; Hadamard helps only as a profile-dependent option.
