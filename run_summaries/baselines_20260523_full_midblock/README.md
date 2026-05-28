# Baseline Runs Summary 2026-05-23

Fresh root `runs/` baseline jobs summarized here: `4005710`-`4005714`. Older runs from 2026-05-18 and VQDM logs were intentionally ignored.

Protocol: SD1.5, full `mid_block` target, `Linear + Conv2d`, weight-only comparison, PickScore prompts, DDIM 50 steps, guidance 5.0, 32 validation prompts x 3 seeds = 96 images.

## Headline Results
| method | label | declared bpp | actual bpp | score | CLIP i-i | LPIPS | CLIP t-i | PSNR | SSIM | quant s | calib s | total s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BitsFusion-lite | bitsfusion_lite_t4p15bpp_bits1x2x3x4 | 4.1500 | 1.3701 | 0.4188 | 0.7459 | 0.3271 | 0.3171 | 16.83 | 0.7611 | 4.8 | 62.9 | 238.5 |
| Scalar PTQ | scalar_ptq_w4 | 4.0000 | 4.0053 | 0.6708 | 0.8577 | 0.1869 | 0.3184 | 20.37 | 0.8689 | 0.2 | 0.0 | 236.4 |
| Q-Diffusion-lite | qdiff_lite_w4 | 4.0000 | 4.0053 | 0.7221 | 0.8722 | 0.1501 | 0.3224 | 21.64 | 0.8967 | 1.2 | 58.1 | 235.0 |
| PTQ4DM-lite | ptq4dm_lite_w4 | 4.0000 | 4.0053 | 0.7194 | 0.8764 | 0.1570 | 0.3181 | 21.46 | 0.8918 | 1.1 | 62.7 | 235.1 |
| AQUATIC-Diff-like | aquatic_diff_lite_bpp4_m4_k256_d8 | 4.0000 | 4.0243 | 0.6992 | 0.8653 | 0.1660 | 0.3205 | 20.99 | 0.8883 | 6.9 | 73.0 | 243.5 |
| BitsFusion-lite | bitsfusion_lite_t3p15bpp_bits1x2x3x4 | 3.1500 | 1.3701 | 0.4181 | 0.7455 | 0.3274 | 0.3171 | 16.83 | 0.7610 | 4.8 | 62.9 | 233.6 |
| Scalar PTQ | scalar_ptq_w3 | 3.0000 | 3.0053 | 0.4319 | 0.7548 | 0.3229 | 0.3168 | 16.91 | 0.7628 | 0.0 | 0.0 | 226.9 |
| Q-Diffusion-lite | qdiff_lite_w3 | 3.0000 | 3.0053 | 0.5682 | 0.8069 | 0.2387 | 0.3176 | 18.83 | 0.8329 | 1.2 | 58.1 | 229.8 |
| PTQ4DM-lite | ptq4dm_lite_w3 | 3.0000 | 3.0053 | 0.5617 | 0.7987 | 0.2371 | 0.3185 | 18.69 | 0.8280 | 1.1 | 62.7 | 230.0 |
| AQUATIC-Diff-like | aquatic_diff_lite_bpp3_m3_k256_d8 | 3.0000 | 3.0182 | 0.6011 | 0.8283 | 0.2272 | 0.3233 | 18.86 | 0.8415 | 3.0 | 73.0 | 231.8 |
| BitsFusion-lite | bitsfusion_lite_t2p15bpp_bits1x2x3x4 | 2.1500 | 1.3701 | 0.4181 | 0.7455 | 0.3274 | 0.3171 | 16.83 | 0.7610 | 4.8 | 62.9 | 233.6 |
| Scalar PTQ | scalar_ptq_w2 | 2.0000 | 2.0053 | 0.1325 | 0.6050 | 0.4724 | 0.2750 | 14.25 | 0.6117 | 0.0 | 0.0 | 227.0 |
| Q-Diffusion-lite | qdiff_lite_w2 | 2.0000 | 2.0053 | 0.2472 | 0.6681 | 0.4209 | 0.2981 | 15.27 | 0.6796 | 1.2 | 58.1 | 230.0 |
| PTQ4DM-lite | ptq4dm_lite_w2 | 2.0000 | 2.0053 | 0.2559 | 0.6850 | 0.4290 | 0.3058 | 15.33 | 0.6819 | 1.1 | 62.7 | 230.0 |
| AQUATIC-Diff-like | aquatic_diff_lite_bpp2_m2_k256_d8 | 2.0000 | 2.0122 | 0.4705 | 0.7723 | 0.3017 | 0.3131 | 17.07 | 0.7704 | 2.0 | 73.0 | 230.7 |
| BitsFusion-lite | bitsfusion_lite_t1p99bpp_bits1x2x3x4 | 1.9900 | 1.3701 | 0.4181 | 0.7455 | 0.3274 | 0.3171 | 16.83 | 0.7610 | 4.8 | 62.9 | 233.6 |
| AQUATIC-Diff-like | aquatic_diff_lite_bpp1p5_m2_k64_d8 | 1.5000 | 1.5030 | 0.3477 | 0.7151 | 0.3675 | 0.2974 | 16.23 | 0.7269 | 1.0 | 73.0 | 229.5 |

## Best By Nominal BPP
| bpp_family_by_declared | best_score_method | best_score_label | best_score | best_score_actual_bpp | best_clip_ii_method | best_clip_ii_label | best_clip_ii | best_clip_ii_actual_bpp | rows_in_family |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 4-ish | Q-Diffusion-lite | qdiff_lite_w4 | 0.7221 | 4.0053 | PTQ4DM-lite | ptq4dm_lite_w4 | 0.8764 | 4.0053 | 5 |
| 3-ish | AQUATIC-Diff-like | aquatic_diff_lite_bpp3_m3_k256_d8 | 0.6011 | 3.0182 | AQUATIC-Diff-like | aquatic_diff_lite_bpp3_m3_k256_d8 | 0.8283 | 3.0182 | 5 |
| 2-ish | AQUATIC-Diff-like | aquatic_diff_lite_bpp2_m2_k256_d8 | 0.4705 | 2.0122 | AQUATIC-Diff-like | aquatic_diff_lite_bpp2_m2_k256_d8 | 0.7723 | 2.0122 | 6 |
| 1.5-ish/low | AQUATIC-Diff-like | aquatic_diff_lite_bpp1p5_m2_k64_d8 | 0.3477 | 1.5030 | AQUATIC-Diff-like | aquatic_diff_lite_bpp1p5_m2_k64_d8 | 0.7151 | 1.5030 | 1 |

## Best By Actual BPP Window
| actual_bpp_window | lo | hi | best_score_method | best_score_label | best_score | best_actual_bpp | rows_in_window |
| --- | --- | --- | --- | --- | --- | --- | --- |
| actual_4bpp_window | 3.5000 | 4.5000 | Q-Diffusion-lite | qdiff_lite_w4 | 0.7221 | 4.0053 | 4 |
| actual_3bpp_window | 2.5000 | 3.5000 | AQUATIC-Diff-like | aquatic_diff_lite_bpp3_m3_k256_d8 | 0.6011 | 3.0182 | 4 |
| actual_2bpp_window | 1.7500 | 2.5000 | AQUATIC-Diff-like | aquatic_diff_lite_bpp2_m2_k256_d8 | 0.4705 | 2.0122 | 4 |
| actual_1_1p75bpp_window | 1.0000 | 1.7500 | BitsFusion-lite | bitsfusion_lite_t4p15bpp_bits1x2x3x4 | 0.4188 | 1.3701 | 5 |

## Ranking By Score
| method | label | declared_target_bpp | actual_bpp | score | clip_ii | lpips | clip_ti | psnr | ssim |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Q-Diffusion-lite | qdiff_lite_w4 | 4.0000 | 4.0053 | 0.7221 | 0.8722 | 0.1501 | 0.3224 | 21.64 | 0.8967 |
| PTQ4DM-lite | ptq4dm_lite_w4 | 4.0000 | 4.0053 | 0.7194 | 0.8764 | 0.1570 | 0.3181 | 21.46 | 0.8918 |
| AQUATIC-Diff-like | aquatic_diff_lite_bpp4_m4_k256_d8 | 4.0000 | 4.0243 | 0.6992 | 0.8653 | 0.1660 | 0.3205 | 20.99 | 0.8883 |
| Scalar PTQ | scalar_ptq_w4 | 4.0000 | 4.0053 | 0.6708 | 0.8577 | 0.1869 | 0.3184 | 20.37 | 0.8689 |
| AQUATIC-Diff-like | aquatic_diff_lite_bpp3_m3_k256_d8 | 3.0000 | 3.0182 | 0.6011 | 0.8283 | 0.2272 | 0.3233 | 18.86 | 0.8415 |
| Q-Diffusion-lite | qdiff_lite_w3 | 3.0000 | 3.0053 | 0.5682 | 0.8069 | 0.2387 | 0.3176 | 18.83 | 0.8329 |
| PTQ4DM-lite | ptq4dm_lite_w3 | 3.0000 | 3.0053 | 0.5617 | 0.7987 | 0.2371 | 0.3185 | 18.69 | 0.8280 |
| AQUATIC-Diff-like | aquatic_diff_lite_bpp2_m2_k256_d8 | 2.0000 | 2.0122 | 0.4705 | 0.7723 | 0.3017 | 0.3131 | 17.07 | 0.7704 |
| Scalar PTQ | scalar_ptq_w3 | 3.0000 | 3.0053 | 0.4319 | 0.7548 | 0.3229 | 0.3168 | 16.91 | 0.7628 |
| BitsFusion-lite | bitsfusion_lite_t4p15bpp_bits1x2x3x4 | 4.1500 | 1.3701 | 0.4188 | 0.7459 | 0.3271 | 0.3171 | 16.83 | 0.7611 |
| BitsFusion-lite | bitsfusion_lite_t3p15bpp_bits1x2x3x4 | 3.1500 | 1.3701 | 0.4181 | 0.7455 | 0.3274 | 0.3171 | 16.83 | 0.7610 |
| BitsFusion-lite | bitsfusion_lite_t2p15bpp_bits1x2x3x4 | 2.1500 | 1.3701 | 0.4181 | 0.7455 | 0.3274 | 0.3171 | 16.83 | 0.7610 |
| BitsFusion-lite | bitsfusion_lite_t1p99bpp_bits1x2x3x4 | 1.9900 | 1.3701 | 0.4181 | 0.7455 | 0.3274 | 0.3171 | 16.83 | 0.7610 |
| AQUATIC-Diff-like | aquatic_diff_lite_bpp1p5_m2_k64_d8 | 1.5000 | 1.5030 | 0.3477 | 0.7151 | 0.3675 | 0.2974 | 16.23 | 0.7269 |
| PTQ4DM-lite | ptq4dm_lite_w2 | 2.0000 | 2.0053 | 0.2559 | 0.6850 | 0.4290 | 0.3058 | 15.33 | 0.6819 |
| Q-Diffusion-lite | qdiff_lite_w2 | 2.0000 | 2.0053 | 0.2472 | 0.6681 | 0.4209 | 0.2981 | 15.27 | 0.6796 |
| Scalar PTQ | scalar_ptq_w2 | 2.0000 | 2.0053 | 0.1325 | 0.6050 | 0.4724 | 0.2750 | 14.25 | 0.6117 |

## Notes
- Strongest 4-ish nominal result by score: Q-Diffusion-lite w4, score 0.7221; PTQ4DM-lite w4 is very close and has the highest CLIP i-i in that family.
- Strongest 3-ish and 2-ish nominal results by score: AQUATIC-Diff-like, with scores 0.6011 and 0.4705.
- Scalar PTQ drops hardest as bpp decreases, which is useful as a lower-bound baseline.
- Warning: BitsFusion-lite produced the same actual bpp for all declared targets: 1.370144. Do not present it as separate 4/3/2 bpp evidence until the allocation behavior is fixed or explicitly justified.
- HPS remains NaN in these files; the meaningful metrics here are CLIP i-i, CLIP t-i, LPIPS, PSNR, SSIM, forward drift, bpp, and timing.

## Files
- `baseline_all_results.csv`
- `baseline_headline_table.csv`
- `baseline_best_by_nominal_bpp.csv`
- `baseline_best_by_actual_bpp_window.csv`
- `source_manifest.json`
- copied compact source summary: `scalar_ptq_summary.csv`
- copied compact source summary: `qdiffusion_lite_summary.csv`
- copied compact source summary: `ptq4dm_lite_summary.csv`
- copied compact source summary: `bitsfusion_lite_summary.csv`
- copied compact source summary: `aquatic_diff_lite_summary.csv`
