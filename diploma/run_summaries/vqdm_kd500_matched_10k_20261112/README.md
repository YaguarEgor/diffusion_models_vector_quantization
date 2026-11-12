# Codebook-only output distillation of DiT: matched10k

Completed scientific snapshot verified12 November2026. Same initial quantized
model, fixed assignments/scales/other parameters,500 Adam updates of codebooks.
All models: DiT-XL/2, ImageNet256, DDPM50, CFG1.5, MSE VAE,10k,
paired records/noise, FP32/TF32 off, official ADM.

| Model | Whole bpp | FID | sFID | IS | Precision | Recall |
|---|---:|---:|---:|---:|---:|---:|
| FP |32|6.097989|21.675457|253.826981|0.7904|0.6659|
| Layerwise-only VQDM |4.150078|15.401847|27.600947|145.312317|0.6491|0.7115|
| Same VQDM+KD500 |4.150078|7.814155|23.367293|211.549194|0.7427|0.6770|

[Exact numbers, protocol and hashes](results.json) and [CSV](results.csv).
Public algorithm cores: [additive representation](../../dit_loss_recode/core.py)
and [output objective/deployment representation](../../dit_objective_followup/objective_core.py).
The snapshot verifies stored measurements, not a self-contained release of
trained weights or the ImageNet dataset. Calibration uses4k real latents and
1k reserved latents from two fixed training shards; a fixed96-example probe
at six timesteps selects checkpoints. It is not a data-free experiment.

The hypothesis is supported in this fixed setting. Recall decreases and no
across-seed interval or compute-matched alternative is established.
VQDM already contains global distillation for U-Net; this is a restricted DiT
transfer experiment, not a novel KD algorithm. Article and practicum talks:
[my_first_article](https://github.com/YaguarEgor/my_first_article).
