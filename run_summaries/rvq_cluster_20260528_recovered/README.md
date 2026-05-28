# RVQ Cluster Logs, 2026-05-28

The uploaded `runs` folder did not contain RVQ summary directories. The fixed-transform RVQ code printed final tables to stdout, but did not write a separate CSV/JSON summary for these runs. This folder reconstructs compact CSV summaries from the `.log` files.

## Files

| file | contents |
|---|---|
| `run_status.csv` | Status of the new `rvq_like-4049*.log` jobs. |
| `rvq_completed_recovered_sweep_results.csv` | All recovered rows from fully completed RVQ jobs. |
| `rvq_completed_best_by_scope_target.csv` | Best completed RVQ row per scope and target bpp. |
| `rvq_completed_best_by_scope_mode_target.csv` | Best completed RVQ row per scope, Hadamard mode, and target bpp. |
| `rvq_interrupted_recovered_sweep_results.csv` | Rows recovered from jobs that ended before final summary. |
| `rvq_all_recovered_sweep_results.csv` | Completed plus interrupted recovered sweep rows. |
| `rvq_recovered_top20_all_rows.csv` | Top 20 recovered rows by score, including interrupted rows. |
| `rvq_vs_vqdm18_compact.csv` | Compact comparison of completed RVQ all-midblock rows against completed VQDM 18-layer baseline. |

## Completed Runs

Only the 18-layer all-midblock jobs for `noH`, `H8`, and `H16` fully completed. The `H32` 18-layer job and all 16-layer jobs ended before final summary export.

| job | scope | Hadamard | rows | status | runtime |
|---:|---|---|---:|---|---:|
| 4049197 | all 18 midblock layers | noH | 13 / 13 | complete, recovered from log | 4h 43m 38s |
| 4049198 | all 18 midblock layers | H8 | 13 / 13 | complete, recovered from log | 4h 51m 43s |
| 4049199 | all 18 midblock layers | H16 | 13 / 13 | complete, recovered from log | 4h 24m 15s |
| 4049200 | all 18 midblock layers | H32 | 10 / 13 | ended without final summary | - |
| 4049201 | VQDM-style 16 layers | noH | 7 / 13 | ended without final summary | - |
| 4049202 | VQDM-style 16 layers | H8 | 8 / 13 | ended without final summary | - |
| 4049203 | VQDM-style 16 layers | H16 | 8 / 13 | ended without final summary | - |
| 4049204 | VQDM-style 16 layers | H32 | 11 / 13 | ended without final summary | - |

## Best Completed RVQ Rows

These are the rows that are safe to use as completed RVQ results.

| scope | target bpp | best mode | stage Ks | index bpp | score | LPIPS | CLIP i-i |
|---|---:|---|---|---:|---:|---:|---:|
| all 18 layers | 4.15 | H16 | 512x512x128x128 | 4.00 | 0.786271 | 0.1247 | 0.9110 |
| all 18 layers | 3.15 | H16 | 1024x128x128 | 3.00 | 0.696052 | 0.1785 | 0.8745 |
| all 18 layers | 2.15 | noH | 64x1024 | 2.00 | 0.564430 | 0.2543 | 0.8188 |

## Comparison To VQDM 18

VQDM 18-layer baseline from job `4036922` is still stronger, but the gap is now modest:

| target bpp | RVQ best score | VQDM 18 score | delta |
|---:|---:|---:|---:|
| 4.15 | 0.786271 | 0.8253 | -0.0390 |
| 3.15 | 0.696052 | 0.7210 | -0.0249 |
| 2.15 | 0.564430 | 0.5865 | -0.0221 |

Against the lighter/scalar baselines, completed RVQ is strong: it beats the full-midblock Vanilla VQ, BitsFusion-lite, Q-Diffusion-lite, PTQ4DM-lite, and scalar PTQ rows at comparable 4/3/2 bpp in the stored summaries. The remaining gap is mainly to VQDM.

## Notes

The missing summary directory is a code/export issue, not evidence that the three completed all-midblock jobs failed. Their final tables are present in stdout logs and were recovered here. Future RVQ runs should write a summary CSV/JSON directly at the end of the fixed-transform sweep to avoid this manual recovery step.
