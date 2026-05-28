# RVQ Cluster Logs, 2026-05-27

Compact ingest of the current root `runs` folder.

## Files

| file | contents |
|---|---|
| `run_status.csv` | Status for every copied `rvq_like-*.log`. |
| `rvq_partial_sweep_results.csv` | All intermediate RVQ `SWEEP RESULT` rows parsed from partial logs. |
| `rvq_partial_best_by_scope_target.csv` | Best partial RVQ row per target scope and bpp. |
| `rvq_partial_best_by_job.csv` | Best partial RVQ row per job. |
| `rvq_partial_top12.csv` | Top 12 partial RVQ rows by score. |
| `completed_runs_reused.csv` | Older complete baseline runs present in `runs`, already summarized in `baselines_20260525_completed`. |

## Completed Runs

No new RVQ run has a final `Run finished`/final summary marker in the copied logs. Since the cluster jobs have already ended, these runs should be treated as interrupted/cancelled before final summary export, not as still pending.

The complete non-RVQ runs present in `runs` are the same baseline runs already summarized in `run_summaries/baselines_20260525_completed`:

| job | method | summary |
|---:|---|---|
| 4038092 | VQDM | `baselines_20260525_completed/vqdm_4038092_summary.csv` |
| 4038093 | Vanilla-VQ | `baselines_20260525_completed/vanilla_vq_4038093_summary.csv` |
| 4038094 | AQUATIC-Diff-lite, 18 layers | `baselines_20260525_completed/aquatic_diff_lite_4038094_all18_summary.csv` |
| 4038095 | AQUATIC-Diff-lite, 16 layers | `baselines_20260525_completed/aquatic_diff_lite_4038095_vqdm16_summary.csv` |

## RVQ Partial Status

Each RVQ stage-wise-K sweep should have 13 sweep rows. None reached a final summary in the copied logs.

| job | scope | Hadamard | sweep rows | status |
|---:|---|---|---:|---|
| 4038138 | all 18 midblock layers | noH | 7 / 13 | ended without final summary |
| 4038139 | all 18 midblock layers | H8 | 8 / 13 | ended without final summary |
| 4038142 | all 18 midblock layers | H16 | 9 / 13 | ended without final summary |
| 4038143 | all 18 midblock layers | H32 | 10 / 13 | ended without final summary |
| 4038145 | VQDM-style 16 layers | noH | 7 / 13 | ended without final summary |
| 4038146 | VQDM-style 16 layers | H8 | 12 / 13 | ended without final summary |
| 4038147 | VQDM-style 16 layers | H16 | 11 / 13 | ended without final summary |
| 4038148 | VQDM-style 16 layers | H32 | 0 / 13 | ended before first sweep result |

## Best Partial Rows

These are not final completed-run numbers, but they are useful progress diagnostics.

| scope | target bpp | best mode | stage Ks | score | LPIPS | CLIP i-i |
|---|---:|---|---|---:|---:|---:|
| all 18 layers | 4.15 | H32 | 512x512x128x128 | 0.780892 | 0.1262 | 0.9071 |
| all 18 layers | 3.15 | H32 | 128x256x512 | 0.703806 | 0.1753 | 0.8791 |
| all 18 layers | 2.15 | H32 | 64x1024 | 0.564214 | 0.2585 | 0.8227 |
| 16 layers | 4.15 | noH | 256x256x256x256 | 0.784801 | 0.1219 | 0.9067 |
| 16 layers | 3.15 | H8 | 256x256x256 | 0.705878 | 0.1720 | 0.8779 |
| 16 layers | 2.15 | H16 | 512x128 | 0.550737 | 0.2625 | 0.8132 |

## Takeaways

The newly copied RVQ logs ended without final summaries, so they should not be used as final comparison rows. Still, the interrupted rows are useful diagnostics: on the 18-layer scope, H32 is best among the rows that appeared; on the 16-layer scope, noH is best at 4.15 bpp, while H8/H16 are better at 3.15/2.15 bpp.

The strongest partial RVQ rows are still below the completed VQDM 16-layer baseline from job 4038092, especially at 3 bpp and 2 bpp. They are far above Vanilla-VQ/AQUATIC-lite at 4 bpp, but the low-bpp gap to VQDM remains the important one.
