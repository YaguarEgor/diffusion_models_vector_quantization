# Baseline Scores For Diffusion Quantization

Рабочая сводка по числам из статей. Важно: эти метрики нельзя напрямую сравнивать с нашими `CLIP image-image`, `LPIPS` и `score = CLIP-I - LPIPS`, потому что в статьях обычно используются FID/IS/CLIP text-image/PickScore на других датасетах, моделях и evaluators.

## Наш текущий ориентир

Текущие лучшие RVQ строки из `run_summaries/rvq_jobb_20260514_005518_core_best_by_target.csv`:

| setting | bpp | score | LPIPS | CLIP image-image | CLIP text-image |
|---|---:|---:|---:|---:|---:|
| RVQ `[64,1024]`, noH, local=0 | 2.00 index / 2.044 effective | 0.6100 | 0.2300 | 0.8400 | ~0.307 |
| RVQ `[512,256,128]`, noH, local=0 | 3.00 index / 3.036 effective | 0.7236 | 0.1587 | 0.8823 | ~0.306 |

Сравнивать это с paper CLIP/FID можно только качественно. Для честного сравнения нужно прогнать наш метод на том же benchmark protocol.

## VQDM

Source: https://arxiv.org/abs/2409.00492, https://ar5iv.labs.arxiv.org/html/2409.00492v1

Model/dataset: SDXL, COCO2014 validation prompts. Metrics: PickScore, CLIP text-image, FID. Weight-only vector quantization of U-Net linear/conv layers.

| method | avg bits | PickScore | CLIP | FID |
|---|---:|---:|---:|---:|
| SDXL original | 32 | 0.226 | 0.357 | 18.99 |
| VQDM | 4.15 | 0.226 | 0.356 | 19.11 |
| VQDM | 3.15 | 0.225 | 0.355 | 19.18 |
| VQDM | 2.15 | 0.219 | 0.341 | 22.14 |
| Q-Diffusion baseline | 4 | 0.225 | 0.355 | 19.30 |
| PTQ4DM baseline | 4 | 0.224 | 0.353 | 19.78 |

VQDM without finetuning:

| method | avg bits | PickScore | CLIP | FID | SbS % |
|---|---:|---:|---:|---:|---:|
| VQDM w/o FT | 4.15 | 0.225 | 0.355 | 20.24 | 34.2 |
| VQDM w/o FT | 3.15 | 0.221 | 0.347 | 24.88 | - |
| VQDM w/o FT | 2.15 | 0.206 | 0.287 | 86.88 | - |

Interpretation:
- VQDM with finetuning is a hard target.
- VQDM without finetuning at 2.15 is much weaker and conceptually closer to our no-finetuning setting.
- They evaluate SDXL, not SD1.5, so this is a conceptual baseline, not a direct scoreboard.

## BitsFusion

Source: https://arxiv.org/abs/2406.04333, https://ar5iv.labs.arxiv.org/html/2406.04333v2

Model/dataset: Stable Diffusion v1.5 UNet, primarily PartiPrompts/MS-COCO/TIFA/GenEval. This is QAT/distillation-style training, not pure PTQ.

PartiPrompts, 1K prompts, CLIP text-image:

| method | bit-width | CLIP |
|---|---:|---:|
| SD-v1.5 | 32 | 0.3175 |
| LSQ | 2 | 0.2849 |
| Q-Diffusion | 4 | 0.3137 |
| EfficientDM | 2 | 0.2918 |
| Apple-MBP | 2 | 0.3023 |
| BitsFusion | 1.99 | 0.3212 |

BitsFusion ablation, average CLIP over CFG 3.5/5.5/7.5/9.5:

| method | bit-width | avg CLIP |
|---|---:|---:|
| SD-v1.5 | 32 | 0.3156 |
| QAT-Base | 2 | 0.2797 |
| +Mixed/Caching | 1.99 | 0.3118 |
| +Feat Dist. | 1.99 | 0.3142 |
| +Time Sampling | 1.99 | 0.3156 |
| +Fine-tuning | 1.99 | 0.3183 |

Interpretation:
- BitsFusion is a very hard practical SD1.5 target.
- It uses training/distillation, so it is not the fairest comparison to training-free RVQ.
- It is probably the strongest "real SD1.5 low-bit" reference.

## DGQ

Source: https://arxiv.org/abs/2501.04304, https://ar5iv.org/html/2501.04304v2

Model/dataset: Stable Diffusion v1.4 and SDXL-Turbo. Metrics on MS-COCO and PartiPrompts. This is mainly activation-aware / group quantization, with weight+activation bit settings.

SD v1.4, MS-COCO / PartiPrompts:

| method | W/A | size | MS-COCO IS | MS-COCO FID | MS-COCO CLIP | Parti CLIP |
|---|---:|---:|---:|---:|---:|---:|
| SD v1.4 FP | 32/32 | 3438 MB | 36.52 | 14.44 | 0.298 | 0.293 |
| Q-Diff | 8/8 | 871 MB | 27.65 | 26.12 | 0.273 | 0.275 |
| TFMQ | 8/8 | 871 MB | 32.79 | 18.85 | 0.286 | 0.286 |
| DGQ, groups=16 | 8/8 | 871 MB | 35.22 | 13.15 | 0.297 | 0.292 |
| Q-Diff | 4/8 | 436 MB | 26.52 | 28.06 | 0.269 | 0.271 |
| TFMQ | 4/8 | 436 MB | 30.85 | 19.98 | 0.281 | 0.281 |
| DGQ, groups=8 | 4/8 | 436 MB | 33.91 | 13.28 | 0.294 | 0.289 |
| DGQ, groups=16 | 4/8 | 436 MB | 33.56 | 13.74 | 0.294 | 0.288 |
| DGQ, groups=16 | 4/6 | 436 MB | 22.17 | 43.66 | 0.263 | 0.274 |

SDXL-Turbo:

| method | W/A | size | MS-COCO IS | MS-COCO FID | MS-COCO CLIP | Parti CLIP |
|---|---:|---:|---:|---:|---:|---:|
| SDXL-Turbo FP | 32/32 | 10269 MB | 35.97 | 21.25 | 0.308 | 0.309 |
| DGQ | 8/8 | 2567 MB | 34.79 | 22.46 | 0.299 | 0.294 |
| DGQ | 4/8 | 1284 MB | 28.33 | 29.22 | 0.289 | 0.291 |
| DGQ | 4/6 | 1284 MB | 22.93 | 45.00 | 0.245 | 0.226 |

Interpretation:
- DGQ W4A8 on SD v1.4 is strong but not unbeatable if we evaluate only weight compression and preserve activations.
- DGQ is especially valuable as an outlier/activation baseline, not as a pure weight-VQ baseline.

## Q-Diffusion

Source: https://openaccess.thecvf.com/content/ICCV2023/html/Li_Q-Diffusion_Quantizing_Diffusion_Models_ICCV_2023_paper.html

Q-Diffusion's original paper gives quantitative tables mostly for unconditional CIFAR/LSUN and only qualitative Stable Diffusion examples. For SD text-to-image numbers, use DGQ/VQDM re-evaluations above.

Original Q-Diffusion unconditional DDIM CIFAR-10:

| method | W/A | FID | IS |
|---|---:|---:|---:|
| Full precision | 32/32 | 4.22 | 9.12 |
| Q-Diffusion | 4/32 | 5.09 | 8.78 |
| Q-Diffusion | 4/8 | 4.93 | 9.12 |

Original Q-Diffusion LDM-4 LSUN-Bedrooms:

| method | W/A | FID |
|---|---:|---:|
| Full precision | 32/32 | 2.98 |
| Q-Diffusion | 4/32 | 4.86 |
| Q-Diffusion | 4/8 | 5.32 |

Interpretation:
- Good historical PTQ baseline.
- Not enough for direct SD1.5 text-to-image comparison unless we rerun it ourselves or use numbers from DGQ/BitsFusion/VQDM.

## AQUATIC-Diff

Source: https://openreview.net/pdf?id=XwERWxaqIr

Model/dataset: LDM-4 ImageNet 256x256 and DDIM CIFAR-10. Codebook/additive vector quantization style, but not SD1.5 text-to-image.

LDM-4 ImageNet 256x256:

| method | W/A | IS | FID | sFID | Precision % |
|---|---:|---:|---:|---:|---:|
| FP | 32/32 | 364.73 | 11.28 | 7.70 | 93.66 |
| Q-Diffusion | 4/8 | 336.80 | 9.29 | 9.29 | 91.06 |
| PTQD | 4/8 | 344.72 | 8.74 | 7.98 | 91.69 |
| EfficientDM | 4/8 | 353.83 | 9.93 | 7.34 | 93.10 |
| AQUATIC-Diff | 4/8 | 356.18 | 9.53 | 5.77 | 93.33 |
| AQUATIC-Diff | 3/8 | 333.57 | 7.84 | 5.81 | 92.39 |
| Q-Diffusion | 2/8 | 49.08 | 43.36 | 17.15 | 43.18 |
| PTQD | 2/8 | 53.36 | 39.37 | 15.14 | 45.89 |
| EfficientDM | 2/8 | 175.03 | 7.60 | 8.12 | 78.90 |
| AQUATIC-Diff | 2/8 | 213.44 | 6.47 | 7.79 | 92.39 |
| AQUATIC-Diff | 1.5/8 | 174.24 | 8.98 | 8.84 | 78.08 |

Efficiency table:

| method | W/A | wall time h | IS | FID | sFID | Precision % |
|---|---:|---:|---:|---:|---:|---:|
| EfficientDM | 4/8 | 3.05 | 353.83 | 9.93 | 7.34 | 93.10 |
| AQUATIC-Diff | 4/8 | 5.61 | 350.28 | 9.04 | 5.77 | 92.80 |
| EfficientDM | 2/8 | 3.11 | 175.03 | 7.60 | 8.12 | 78.90 |
| AQUATIC-Diff | 2/8 | 6.95 | 180.57 | 8.10 | 9.58 | 78.87 |

Interpretation:
- Very relevant conceptually because it is additive/codebook quantization for diffusion.
- Not a direct SD1.5 baseline.
- Hard to beat at 2-bit if we move to the same LDM-4/ImageNet protocol.

## VQ4DiT

Source: https://arxiv.org/abs/2408.17131, https://ar5iv.org/html/2408.17131v1

Model/dataset: DiT XL/2, ImageNet 256/512. Vector quantization for Diffusion Transformers, not U-Net SD1.5.

ImageNet 256x256, DiT XL/2:

| steps | method | bits | size MB | FID | sFID | IS | Precision |
|---:|---|---:|---:|---:|---:|---:|---:|
| 50 | FP | 32 | 2553.35 | 6.72 | 21.13 | 243.90 | 0.7848 |
| 50 | VQ4DiT | 3 | 241.14 | 11.91 | 24.18 | 263.93 | 0.9096 |
| 50 | VQ4DiT | 2 | 162.08 | 12.42 | 25.16 | 209.95 | 0.8725 |
| 100 | FP | 32 | 2553.35 | 5.59 | 18.63 | 269.67 | 0.8156 |
| 100 | VQ4DiT | 3 | 241.14 | 10.74 | 23.90 | 265.63 | 0.9018 |
| 100 | VQ4DiT | 2 | 162.08 | 11.85 | 23.64 | 213.77 | 0.8836 |

Interpretation:
- Not a baseline for SD1.5 U-Net.
- Very useful justification for recoding/assignment calibration.

## Who can we plausibly beat soon?

Likely easiest targets:

1. Q-Diffusion-style scalar PTQ at low bit, especially when evaluated as weight-only or without activation quantization.
2. VQDM without finetuning at 2.15, if we reproduce a close enough setup.
3. Some 4-bit scalar baselines on our own SD1.5 midblock/UNet protocol, if our RVQ remains stable when expanded beyond the tested layer set.

Hard targets:

1. VQDM with finetuning, especially 3.15 and 4.15.
2. BitsFusion 1.99, because it is SD1.5-specific and uses training/distillation.
3. AQUATIC-Diff 2/8 on LDM-4/ImageNet, because it is codebook/additive and already strong.
4. DGQ W4A8 if we compare full weight+activation deployment quality, because it explicitly handles activation outliers and attention.

Practical conclusion:

For the course work, the cleanest claim is not "we beat all baselines", but:

- RVQ is a much cheaper no-finetuning alternative to AQ/VQDM-style vector quantization.
- Stage-wise K is the most promising knob under fixed packed bpp.
- Full-XTX recoding is justified by VQ assignment-calibration literature.
- To compete with BitsFusion/DGQ/VQDM-FT, the next algorithmic step should probably be outlier/channel scaling, timestep-aware XTX, and eventually data-aware/learnable rotation.

