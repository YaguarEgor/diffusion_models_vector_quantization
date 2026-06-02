# RVQ Final Sweep Summary, 2026-05-31

Compact final RVQ stage-wise-K sweep summary. Only completed rows are stored in the final tables.

This folder combines:

- completed new runs from root `runs`: `4063459`-`4063463`;
- previously completed all-midblock runs recovered in `rvq_cluster_20260528_recovered`: `4049197`-`4049199`.

## Files

| file | contents |
|---|---|
| `rvq_new_completed_sweep_results.csv` | All completed rows from new jobs `4063459`-`4063463`; 65 rows. |
| `rvq_final_completed_sweep_results.csv` | Complete 8-run RVQ set; 104 rows. |
| `rvq_final_best_by_scope_mode_target.csv` | Best row for each scope, Hadamard mode, and target bpp. |
| `rvq_final_best_by_scope_target.csv` | Best row for each scope and target bpp across Hadamard modes. |
| `rvq_final_headline.csv` | Small headline table for paper/report use. |

## Headline Best Rows

| scope | target bpp | best H | stage Ks | index bpp | score | CLIP i-i | LPIPS | job |
|---|---:|---|---|---:|---:|---:|---:|---:|
| all 18 layers | 4.15 | H16 | 512x512x128x128 | 4.00 | 0.786271 | 0.911018 | 0.124747 | 4049199 |
| all 18 layers | 3.15 | H32 | 128x256x512 | 3.00 | 0.703806 | 0.879102 | 0.175296 | 4063459 |
| all 18 layers | 2.15 | noH | 64x1024 | 2.00 | 0.564430 | 0.818766 | 0.254336 | 4049197 |
| 16 layers | 4.15 | H32 | 512x512x128x128 | 4.00 | 0.781065 | 0.907035 | 0.125971 | 4063463 |
| 16 layers | 3.15 | H32 | 128x256x512 | 3.00 | 0.704430 | 0.879481 | 0.175051 | 4063463 |
| 16 layers | 2.15 | H8 | 64x1024 | 2.00 | 0.587828 | 0.833258 | 0.245431 | 4063461 |

## Short Interpretation

- The missing final RVQ set is now complete: all 8 planned scope/H-mode runs have 13/13 rows when combined with the already completed `4049197`-`4049199` runs.
- For 18-layer full midblock, H32 improves the 3.15-bpp point, but H16 remains best at 4.15 and noH remains best at 2.15.
- For 16-layer VQDM-style scope, H32 is best at 4.15 and 3.15, while H8 is best at 2.15.
- The 16-layer scope is slightly better than 18-layer at 2.15 and 3.15, but slightly worse at 4.15.
- Against stored VQDM baselines, RVQ is still below: for 18 layers by about `-0.0390/-0.0172/-0.0221` score at 4.15/3.15/2.15 bpp, and for 16 layers by about `-0.0191/-0.0465/-0.0299`.
