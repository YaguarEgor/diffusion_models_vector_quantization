# Baseline Results, 2026-05-25

Compact summaries from the root `runs` folder. Raw logs and JSON row dumps are not copied here.

## Files

| file | contents |
|---|---|
| `completed_baselines_compact.csv` | Normalized compact table across all completed baseline runs. |
| `vqdm_4038092_summary.csv` | Original compact VQDM summary CSV. |
| `vanilla_vq_4038093_summary.csv` | Original compact Vanilla-VQ summary CSV. |
| `aquatic_diff_lite_4038094_all18_summary.csv` | Original compact AQUATIC-Diff-lite 18-layer summary CSV. |
| `aquatic_diff_lite_4038095_vqdm16_summary.csv` | Original compact AQUATIC-Diff-lite 16-layer summary CSV. |

## Completed Runs

| job | method | scope | rows | wall time |
|---:|---|---|---:|---:|
| 4038092 | VQDM | 16 VQDM-style midblock+conv layers | 3 | 1h 42m 55s |
| 4038093 | Vanilla-VQ | 16 VQDM-style midblock+conv layers | 5 | 25m 11s |
| 4038094 | AQUATIC-Diff-lite | 18 full midblock+conv layers | 4 | 19m 26s |
| 4038095 | AQUATIC-Diff-lite | 16 VQDM-style midblock+conv layers | 4 | 21m 07s |

No fatal `Traceback`, `RuntimeError`, OOM, Slurm cancel, or killed patterns were found in the copied stderr files.

## Main Takeaways

VQDM is still the strongest completed vector baseline in this batch. At roughly 4/3/2 bpp it gets score 0.800/0.751/0.618 with CLIP i-i 0.919/0.901/0.847.

Vanilla-VQ is a deliberately lower VQ baseline. It reaches 0.717 at 4 bpp, 0.617 at 3 bpp, and then drops sharply to 0.456 at 2 bpp.

AQUATIC-Diff-lite is below VQDM here. The 16-layer run is slightly better than the 18-layer run at 4 and 2 bpp, but the difference is small enough that it should be treated as a scope/eval comparison, not a strong algorithmic conclusion.

Among completed baselines, the target to beat remains VQDM. Final RVQ comparisons are tracked separately in `../rvq_final_20260531/`.
