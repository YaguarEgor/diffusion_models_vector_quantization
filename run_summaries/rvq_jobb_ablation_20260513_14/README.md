# RVQ JobB Ablation Summaries

This folder consolidates the JobB-style RVQ ablations.

## Contents

| file | contents |
|---|---|
| `jobb_20260513_001931_summary.csv` | Older JobB run with fixed `K` variants and full-XTX recode-pass comparisons. Useful for the recoding section. |
| `jobb_20260513_001931_summary.md` | Short human-readable summary of the older JobB run. |
| `rvq_jobb_20260514_005518_summary.csv` | Latest JobB v2 run: stage-wise `K` profiles, entropy/utilization statistics, local recode on/off. |
| `quant_v2_20260514_005518_summary.md` | Human-readable summary of the latest JobB v2 run. |
| `quant_v2_20260514_005518_metadata.json` | Compact metadata for the latest JobB v2 run. |
| `rvq_jobb_20260514_005518_core_best_by_target.csv` | Best JobB v2 core row per target bpp. |
| `rvq_jobb_20260514_005518_core_best_by_stageKs.csv` | Best JobB v2 core row per stage-wise `K` profile. |
| `rvq_jobb_20260514_005518_core_best_by_target_family_local.csv` | Best rows split by target, transform family, and local recode flag. |
| `rvq_jobb_20260514_005518_top20.csv` | Top 20 JobB v2 rows by score. |
| `rvq_jobb_20260514_005518_all_linear.csv` | Additional all-linear rows from JobB v2. |

## Use

- Use `jobb_20260513_001931_summary.csv` for the fixed-`K` recoding experiment.
- Use `rvq_jobb_20260514_005518_summary.csv` and its derived compact tables for stage-wise `K`, entropy/utilization, and local recode analysis.
- These runs are ablations, not final full-midblock headline results. Final RVQ headline results are in `../rvq_final_20260531/`.
