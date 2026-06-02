# RVQ Hadamard Clean Sweep, 2026-06-02

Source run: `rvq_hadamard_clean-4066076`.

This is the isolated Hadamard-block experiment for RVQ:

- target: 10 Linear layers in SD1.5 mid-block;
- no Conv2d layers;
- `d=8`;
- fixed `K=256`;
- profiles: `2.15`, `3.15`, `4.15`;
- no stage-wise K sweep;
- no full-XTX post-RVQ recode;
- transforms: `noH` plus Hadamard blocks `8, 16, 32, 64, 128, 256` with and without random signs;
- evaluation size: 96 generated samples.

The run completed successfully: `39 / 39` variants. The `.err` file contains only a `timm` deprecation warning, no traceback/OOM/cancel error.

## Files

| file | contents |
|---|---|
| `run_status.csv` | compact run metadata |
| `rvq_hadamard_clean_all_results.csv` | full exported CSV from the run |
| `rvq_hadamard_clean_all_results.json` | full exported JSON from the run |
| `rvq_hadamard_clean_compact.csv` | compact numeric table for analysis |
| `rvq_hadamard_clean_best_by_profile.csv` | best row by score for each bpp profile |
| `rvq_hadamard_clean_top5_by_profile.csv` | top-5 rows by score for each bpp profile |
| `rvq_hadamard_clean_delta_vs_noh.csv` | deltas against noH inside each profile |
| `rvq_hadamard_clean_by_block.csv` | best/mean/min/max score for each Hadamard block |

## Best Rows

| profile | best transform | score | LPIPS | CLIP I-I | CLIP T-I | quant build, s |
|---|---|---:|---:|---:|---:|---:|
| `2.15-d8` | `noH` | 0.589017 | 0.240649 | 0.829666 | 0.306556 | 303.99 |
| `3.15-d8` | `H16 signs=False` | 0.706637 | 0.180422 | 0.887059 | 0.310792 | 456.20 |
| `4.15-d8` | `noH` | 0.806061 | 0.112121 | 0.918182 | 0.310185 | 608.94 |

## Positive Hadamard Deltas

Only the `3.15` profile has Hadamard variants that beat `noH` by score:

| profile | transform | score | delta score vs noH | delta LPIPS vs noH | delta CLIP I-I vs noH |
|---|---|---:|---:|---:|---:|
| `3.15-d8` | `H16 signs=False` | 0.706637 | +0.014086 | -0.000107 | +0.013979 |
| `3.15-d8` | `H32 signs=False` | 0.699316 | +0.006765 | -0.000007 | +0.006758 |
| `3.15-d8` | `H8 signs=True` | 0.695437 | +0.002886 | +0.002871 | +0.005757 |

For `2.15` and `4.15`, the best row remains `noH`. On `2.15`, the closest Hadamard row is `H8 signs=False` with score `0.583215`, which is `-0.005802` below `noH`. On `4.15`, the closest Hadamard row is `H8 signs=True` with score `0.797795`, which is `-0.008265` below `noH`.

## Interpretation

This run supports a narrow conclusion: Hadamard is not a uniformly useful default transform for RVQ. In the clean fixed-`K` setting, it helps the medium-bit profile (`3.15`) and the best option there is `H16 signs=False`. At low and high bpp the identity transform wins.

The block-size trend is also clear enough for the course text: small and medium blocks are the only reasonable candidates. Blocks `64`, `128`, and especially `256` usually degrade score. Random signs are not consistently helpful: they improve some variants, but the strongest positive result is `H16 signs=False`.

Practical takeaway: Hadamard should stay as a per-profile hyperparameter, not as an always-on operation. For later full runs, the sensible shortlist is `noH`, `H8`, `H16`, and possibly `H32`; larger blocks are not worth prioritizing.
