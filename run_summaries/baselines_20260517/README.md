# Baseline Cluster Runs, 2026-05-17

Metric convention: `score = CLIP image-image - LPIPS`; higher is better. CLIP i-i, LPIPS, PSNR and SSIM are preserved in the CSVs, so the comparison is not locked to this combined score.

## Run Status

- Scalar PTQ, QDiffusion-lite and PTQ4DM-lite completed normally on 96 eval items.
- VQDM completed `single_largest_linear` and `hardcoded_10_linear` for 4.15/3.15/2.15, plus `hardcoded_10_linear_plus_conv` for 4.15; Slurm cancelled it while continuing plus-conv 3.15.
- RVQ reference below uses previous `all_completed_rvq_funnel_results.csv` on 96 eval items. The newer JobB v2 table is saved too, but it is treated as secondary because its eval timing indicates a smaller eval set.

## Headline: Hardcoded 10 Linear Layers

| Target | RVQ best profile | RVQ strict bpp | Best non-VQDM | VQDM hardcoded | Strict verdict |
| --- | --- | --- | --- | --- | --- |
| 4.15 | 0.8010 @ 4.00bpp | 0.8010 @ 4.00bpp | PTQ4DM-lite 0.7629 | 0.8437 @ 4.05bpp | RVQ strict loses to VQDM |
| 3.15 | 0.7616 @ 3.38bpp | 0.7121 @ 3.00bpp | QDiffusion-lite 0.6097 | 0.7541 @ 3.04bpp | RVQ strict loses to VQDM |
| 2.15 | 0.6262 @ 2.25bpp | 0.5907 @ 2.00bpp | PTQ4DM-lite 0.4134 | 0.6414 @ 2.03bpp | RVQ strict loses to VQDM |

Interpretation: RVQ is clearly above scalar/QDiffusion-lite/PTQ4DM-lite at all three targets. Against VQDM, strict bpp-matched RVQ still loses at 4.15 and 2.15, and also loses at 3.15 if we require `index_bpp <= target_bpp`. The tempting 3.15 RVQ win comes from a K=512 row at 3.38 index bpp, so it should be described as a profile/ablation win, not a clean equal-bpp win.

## Best Baseline Per Family

| Target | Family | Label | bpp | Score | LPIPS | CLIP i-i | CLIP t-i |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 4.15 | VQDM | VQDM target~4.15bpp | 4.053 | 0.8437 | 0.0921 | 0.9358 | 0.3093 |
| 4.15 | Scalar PTQ | scalar_ptq_w4 | 4.011 | 0.7116 | 0.1617 | 0.8733 | 0.3191 |
| 4.15 | QDiffusion-lite | qdiff_lite_w4 | 4.011 | 0.7378 | 0.1422 | 0.8800 | 0.3197 |
| 4.15 | PTQ4DM-lite | ptq4dm_lite_w4 | 4.011 | 0.7629 | 0.1313 | 0.8942 | 0.3187 |
| 3.15 | VQDM | VQDM target~3.15bpp | 3.042 | 0.7541 | 0.1439 | 0.8980 | 0.3070 |
| 3.15 | Scalar PTQ | scalar_ptq_w3a8 | 3.011 | 0.5284 | 0.2652 | 0.7936 | 0.3161 |
| 3.15 | QDiffusion-lite | qdiff_lite_w3 | 3.011 | 0.6097 | 0.2161 | 0.8258 | 0.3210 |
| 3.15 | PTQ4DM-lite | ptq4dm_lite_w3 | 3.011 | 0.6085 | 0.2191 | 0.8276 | 0.3180 |
| 2.15 | VQDM | VQDM target~2.15bpp | 2.032 | 0.6414 | 0.2151 | 0.8566 | 0.3091 |
| 2.15 | Scalar PTQ | scalar_ptq_w2 | 2.011 | 0.1776 | 0.4470 | 0.6245 | 0.2753 |
| 2.15 | QDiffusion-lite | qdiff_lite_w2 | 2.011 | 0.3404 | 0.3765 | 0.7169 | 0.3031 |
| 2.15 | PTQ4DM-lite | ptq4dm_lite_w2 | 2.011 | 0.4134 | 0.3295 | 0.7428 | 0.3141 |

## Files

- `baseline_all_results.csv`
- `baseline_hardcoded10_best_by_family_target.csv`
- `baseline_hardcoded10_best_overall_by_target.csv`
- `rvq_best_hardcoded10_profile_best.csv`
- `rvq_best_hardcoded10_strict_bpp.csv`
- `rvq_jobb_v2_best_hardcoded10.csv`
- `rvq_vs_baselines_hardcoded10_profile_best.csv`
- `rvq_vs_baselines_hardcoded10_strict_bpp.csv`
- `headline_comparison_hardcoded10.csv`
- `source_manifest.json`
