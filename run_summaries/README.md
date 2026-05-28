# Run Summaries Index

This folder contains only compact experiment summaries: CSV tables, Markdown notes, and one LaTeX table. Raw logs, setup-failure notes, full JSON payloads, and source configs are intentionally excluded.

## Main Tables

| file | contents |
|---|---|
| `all_completed_rvq_funnel_results.csv` | Normalized RVQ funnel rows from jobs `3970494` and `3988130`; 32 completed rows. |
| `rvq_completed_results.csv` | Original completed-row CSV for job `3970494`; 14 rows. |
| `rvq_like-3988130_results.csv` | Completed-row CSV for job `3988130`; 18 rows. |
| `vqdm_like-3948839_results.csv` | VQDM baseline rows from `vqdm_like-3948839.log`; 3 rows. |
| `jobb_20260513_001931_summary.csv` | JobB old budget-allocation run summary; 55 rows. |
| `rvq_jobb_20260514_005518_summary.csv` | Latest JobB v2 stage-wise-K run summary; 48 rows. |
| `rvq_jobb_20260514_005518_top20.csv` | Top-20 rows from latest JobB v2 run by `score = CLIP-I - LPIPS`. |
| `rvq_jobb_20260514_005518_core_best_by_target.csv` | Best latest JobB v2 core row per target bpp. |
| `baselines_20260525_completed/completed_baselines_compact.csv` | Completed baseline rows from jobs `4038092`-`4038095`; VQDM, Vanilla-VQ, and AQUATIC-Diff-lite. |
| `rvq_cluster_20260527_status/rvq_partial_sweep_results.csv` | Parsed interrupted RVQ stage-wise-K sweep rows from jobs `4038138`-`4038148`; no final RVQ run completed in these logs. |
| `rvq_cluster_20260528_recovered/rvq_completed_recovered_sweep_results.csv` | Recovered completed RVQ all-midblock rows from jobs `4049197`-`4049199`; summary directories were missing, so rows were parsed from logs. |

## Human-Readable Summaries

| file | contents |
|---|---|
| `rvq_like-3988130_summary.md` | Fresh root `runs` server run summary; K=512 helped, Slurm cancelled the job, only 1 GPU was visible. |
| `rvq_experiment_results_summary.md` | Earlier job `3970494` summary and notes; includes d=8 vs d=16, Hadamard notes, full-XTX recode notes. |
| `rvq_d8_d16_comparison.tex` | LaTeX tables for d=8 vs d=16 comparison from job `3970494`. |
| `vqdm_like-3948839_summary.md` | VQDM baseline summary from the available log. |
| `jobb_20260513_001931_summary.md` | Short summary of the older JobB budget-allocation run. |
| `quant_v2_20260514_005518_summary.md` | Human-readable summary of the latest JobB v2 stage-wise-K run. |
| `quant_v2_20260514_005518_metadata.json` | Compact run metadata for latest JobB v2. |
| `baselines_20260525_completed/README.md` | Status and conclusions for the newly copied completed baseline logs. |
| `rvq_cluster_20260527_status/README.md` | Status and compact interrupted-run summary for the latest copied RVQ cluster logs. |
| `rvq_cluster_20260528_recovered/README.md` | Recovered summary for the latest RVQ cluster logs; includes completed all-midblock rows and interrupted 16-layer/H32 rows. |

## Current Research Conclusions

1. `d=16` is not worth keeping in the main sweep. In job `3970494`, `d=8` wins on all completed paired comparisons and `d=16` is roughly 1.95x slower.
2. Full-XTX post-RVQ recoding should stay on. It improves the full-XTX objective while keeping codebooks fixed, so it is not VQDM-style finetuning.
3. Hadamard is mixed. It helped some low-bpp or K=256 cases, but the newer `3988130` run shows noH winning for K=512 at 2.15 and 3.15 targets.
4. Increasing K is the strongest signal so far. In job `3988130`, K=512 improved low-bpp quality substantially, but it also increases packed index bpp, so stage-wise-K tests are needed for fair budget control.
5. Old budget allocation experiments should be treated as an ablation record, not the main direction. The promising direction is stage-wise K under fixed bpp, plus usage/entropy statistics.
6. Latest JobB v2 confirms stage-wise K is useful under fixed packed index bpp: best 2.15 is `[64,1024]`, best 3.15 is `[512,256,128]`.
7. Local recode after each RVQ stage is not promising in the current form: latest JobB v2 has local recode losing 18/20 paired core comparisons.
8. VQDM baseline from `vqdm_like-3948839.log` is much stronger numerically, but it used n=30 evaluation, while the RVQ funnel rows here mostly use n=96. Compare carefully.

## Key Best Rows

RVQ job `3988130`, completed rows:

| target bpp | best completed RVQ variant | index bpp | score | LPIPS | CLIP i-i |
|---:|---|---:|---:|---:|---:|
| 4.15 | K=256 noH | 4.00 | 0.800970 | 0.1175 | 0.9185 |
| 3.15 | K=512 noH | 3.38 | 0.761580 | 0.1395 | 0.9011 |
| 2.15 | K=512 noH | 2.25 | 0.626237 | 0.2156 | 0.8419 |

Latest JobB v2 `20260514_005518`, stage-wise K under fixed packed bpp:

| target bpp | best variant | index bpp | effective bpp | score | LPIPS | CLIP i-i |
|---:|---|---:|---:|---:|---:|---:|
| 3.15 | `[512,256,128]` noH local=0 | 3.00 | 3.0365 | 0.723620 | 0.1587 | 0.8823 |
| 2.15 | `[64,1024]` noH local=0 | 2.00 | 2.0443 | 0.610001 | 0.2300 | 0.8400 |

VQDM baseline `3948839`:

| target bpp | score | LPIPS | CLIP i-i | n |
|---:|---:|---:|---:|---:|
| 4.15 | 0.957463 | 0.0268 | 0.9843 | 30 |
| 3.15 | 0.924190 | 0.0520 | 0.9762 | 30 |
| 2.15 | 0.882565 | 0.0754 | 0.9579 | 30 |
