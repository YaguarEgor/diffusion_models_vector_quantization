# RVQ Ablation Digest 2026-05-24

Sources:
- `rvq_completed_results.csv`
- `rvq_like-3988130_results.csv`
- `all_completed_rvq_funnel_results.csv`
- `jobb_20260513_001931_summary.csv`
- `rvq_jobb_20260514_005518_summary.csv`
- `rvq_jobb_20260514_005518_top20.csv`
- `quant_v2_20260514_005518_summary.md`
- `baselines_20260517/README.md`
- `baselines_20260517/rvq_best_hardcoded10_profile_best.csv`
- `baselines_20260517/rvq_best_hardcoded10_strict_bpp.csv`
- `baselines_20260517/rvq_jobb_v2_best_hardcoded10.csv`

Metric: `score = CLIP image-image - LPIPS`, unless noted otherwise.

## Main Conclusions

1. `d=16` should stay out of the main runs.
   It was consistently worse than `d=8` and almost doubled quantization time.

2. Broad Hadamard block sweeps are not worth repeating.
   If Hadamard is tested, use only `H8` and maybe `H16` for the old `K=256, 2.15 bpp` setting.

3. Hadamard is not reliably the best variant.
   It sometimes helps `K=256`, especially around low bpp, but the best K/stage-K variants are usually `noH`.

4. Changing K is the strongest useful direction.
   Uniform `K=512` improved quality but increases bpp. Stage-wise K gives better bpp-aligned variants.

5. Full-XTX post-RVQ recode should stay enabled.
   It is not finetuning: codebooks stay fixed, only discrete codes are updated. Earlier logs showed large objective improvement.

6. Local recode after each stage should be disabled.
   In the stage-wise-K run it won only 2/20 paired core comparisons and lost 18/20; mean score delta was `-0.040823`.

7. Multistart was not separately ablated.
   Serious RVQ runs used `kmeans_n_init=4` per stage. There is no table evidence that lowering or increasing it improves final metrics.

8. Against scalar/QDiffusion-lite/PTQ4DM-lite, RVQ is already clearly stronger on the hardcoded-10-linear setup.
   Against VQDM, strict bpp-matched RVQ still loses in the saved comparison tables. The tempting 3.15 win comes from `K=512` at `3.38` index bpp, so it is not a clean equal-bpp win.

9. All conclusions below are candidate-selection conclusions, not final full-midblock claims.
   Most RVQ ablations here are on 10 Linear layers. The latest JobB v2 stage-wise-K run also has Conv disabled. Final 16-layer and 18-layer full-midblock runs are still needed.

## d=8 vs d=16

| target bpp | d | M | score | LPIPS | CLIP i-i | quant s |
|---:|---:|---:|---:|---:|---:|---:|
| 4.15 | 8 | 4 | 0.792739 | 0.1202 | 0.9130 | 599.3 |
| 4.15 | 16 | 8 | 0.786131 | 0.1243 | 0.9104 | 1166.6 |
| 3.15 | 8 | 3 | 0.689224 | 0.1819 | 0.8711 | 443.8 |
| 3.15 | 16 | 6 | 0.657902 | 0.2006 | 0.8585 | 872.3 |
| 2.15 | 8 | 2 | 0.580259 | 0.2481 | 0.8284 | 298.3 |
| 2.15 | 16 | 4 | 0.570669 | 0.2537 | 0.8244 | 582.3 |

Conclusion: `d=8` is both better and faster.

## Hadamard Blocks, K=256

From `rvq_like-3988130_results.csv`.

| target bpp | best variant | score | LPIPS | CLIP i-i | note |
|---:|---|---:|---:|---:|---|
| 4.15 | noH | 0.800970 | 0.1175 | 0.9185 | better than H8/H16 in this run |
| 3.15 | H8 | 0.712077 | 0.1712 | 0.8833 | only slightly above noH 0.710136 |
| 2.15 | H16 | 0.590704 | 0.2413 | 0.8320 | clearly above noH 0.568847 |

Conclusion: if using old uniform `K=256`, the only Hadamard block worth keeping is `H16` for 2 bpp and maybe `H8` as a small ablation. No reason to sweep `8,16,32,64,128,256` again.

## Uniform K Ablation

From `rvq_like-3988130_results.csv`.

| target bpp | K | best variant | index bpp | score | LPIPS | CLIP i-i | quant s |
|---:|---:|---|---:|---:|---:|---:|---:|
| 3.15 | 128 | noH | 3.50 | 0.729444 | 0.1558 | 0.8852 | 320.0 |
| 3.15 | 256 | H8 | 3.00 | 0.712077 | 0.1712 | 0.8833 | 461.9 |
| 3.15 | 512 | noH | 3.38 | 0.761580 | 0.1395 | 0.9011 | 874.0 |
| 2.15 | 128 | H16 | 1.75 | 0.553258 | 0.2653 | 0.8186 | 161.3 |
| 2.15 | 256 | H16 | 2.00 | 0.590704 | 0.2413 | 0.8320 | 304.8 |
| 2.15 | 512 | noH | 2.25 | 0.626237 | 0.2156 | 0.8419 | 582.4 |

Conclusion: increasing `K` helps a lot, but uniform `K=512` is not equal-bpp with the old `K=256` line.

## Old Budget Allocation Run

From `jobb_20260513_001931_summary.md`.

| rank | section | scope | profile | family | bpp | score | LPIPS | CLIP i-i |
|---:|---|---|---|---|---:|---:|---:|---:|
| 1 | core_low_bpp_k_recode_h_noh | hardcoded_linear | 3.15-d8-k512 | H | 3.375 | 0.781023 | 0.1262 | 0.9072 |
| 2 | core_low_bpp_k_recode_h_noh | hardcoded_linear | 3.15-d8-k512 | H | 3.375 | 0.763791 | 0.1361 | 0.8999 |
| 3 | budget_allocation | all_midblock_linear | mixed | H | n/a | 0.763791 | 0.1361 | 0.8999 |

Conclusion: budget allocation did not become the main direction. This run mainly supported the idea that changing `K` is more promising than redistributing a fixed budget in the old way. Keep it as an ablation record, not as the configuration to rerun.

## Stage-Wise K

From `rvq_jobb_20260514_005518_summary.csv`.

| target bpp | best stage Ks | family | local recode | effective bpp | entropy bpp | utilization | score | LPIPS | CLIP i-i |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|
| 3.15 | `[512, 256, 128]` | noH | 0 | 3.0365 | 2.9857 | 1.000 | 0.723620 | 0.1587 | 0.8823 |
| 2.15 | `[64, 1024]` | noH | 0 | 2.0443 | 1.9941 | 1.000 | 0.610001 | 0.2300 | 0.8400 |

Best alternatives:

| target bpp | stage Ks | family | local recode | effective bpp | score |
|---:|---|---|---:|---:|---:|
| 3.15 | `[256, 256, 256]` | noH | 0 | 3.0312 | 0.707586 |
| 3.15 | `[1024, 128, 128]` | noH | 0 | 3.0521 | 0.706585 |
| 2.15 | `[512, 128]` | noH | 0 | 2.0260 | 0.589800 |
| 2.15 | `[256, 256]` | H8 | 0 | 2.0208 | 0.588522 |

Conclusion: the best bpp-aligned low-bpp candidates are `noH + local=0` with uneven K allocation:
- 3 bpp: `[512, 256, 128]`
- 2 bpp: `[64, 1024]`

Important scope note: this JobB v2 run is useful for choosing candidates, but it was not a final full-midblock run. It used `hardcoded_linear` / `all_midblock_linear`, final prompts 32, and Conv stage was disabled (`RVQ_JOBB_CONV_TOPK_PER_GROUP=0`).

## Recode

### Full-XTX post-RVQ recode

Earlier logs reported 150 full-XTX recode calls:
- mean relative objective improvement: `47.53%`
- median relative objective improvement: `50.26%`
- min/max: `21.87% / 58.48%`

Conclusion: keep full-XTX post-RVQ recode always enabled.

### Local recode after each stage

From `quant_v2_20260514_005518_summary.md`:
- local recode won `2/20` paired core comparisons
- lost `18/20`
- mean `local1 - local0` score delta: `-0.040823`

Conclusion: disable local recode in final runs.

## Baseline Cross-Check

From `baselines_20260517/README.md` and the RVQ comparison CSVs. Scope: hardcoded 10 Linear layers.

| target bpp | RVQ profile-best | RVQ strict-bpp | best non-VQDM baseline | VQDM hardcoded | strict verdict |
|---:|---:|---:|---|---:|---|
| 4.15 | 0.8010 @ 4.00 | 0.8010 @ 4.00 | PTQ4DM-lite 0.7629 | 0.8437 @ 4.05 | RVQ loses to VQDM |
| 3.15 | 0.7616 @ 3.38 | 0.7121 @ 3.00 | QDiffusion-lite 0.6097 | 0.7541 @ 3.04 | RVQ loses to VQDM under strict bpp |
| 2.15 | 0.6262 @ 2.25 | 0.5907 @ 2.00 | PTQ4DM-lite 0.4134 | 0.6414 @ 2.03 | RVQ loses to VQDM |

Conclusion: RVQ already beats the scalar-family baselines in these saved tables, but not VQDM when bpp is matched strictly. The `K=512` 3.15 profile-best row is promising but should be described as an ablation win at higher bpp, not as a fair VQDM win.

## Coverage And Gaps

Covered by saved summaries:
- `d=8` vs `d=16`
- Hadamard/noH and selected Hadamard blocks
- off-diagonal XTX diagnostics before/after Hadamard
- uniform K sweep (`K=128/256/512`)
- old budget allocation sweep
- stage-wise K sweep
- full-XTX post-RVQ recode
- local recode after each stage
- entropy/effective-bpp/utilization statistics for stage-wise K
- hardcoded-10-linear comparison against scalar/QDiffusion-lite/PTQ4DM-lite/VQDM

Not covered by a clean saved ablation:
- changing `kmeans_n_init` / multistart count
- final full-midblock 16-layer and 18-layer RVQ results with the selected configs
- final full-midblock Conv-inclusive check for the stage-wise-K candidates

## Recommended Final RVQ Variants

Run these on both 18-layer and 16-layer target scopes:

| purpose | target bpp | config |
|---|---:|---|
| high-bpp anchor | 4.15 | `d=8, K=256, M=4, noH, full-XTX recode=1` |
| main 3-bpp candidate | 3.15 | `d=8, stage_Ks=[512,256,128], noH, local=0, full-XTX recode=1` |
| main 2-bpp candidate | 2.15 | `d=8, stage_Ks=[64,1024], noH, local=0, full-XTX recode=1` |
| optional Hadamard ablation | 3.15 | same as main 3-bpp, but `H8` |
| optional Hadamard ablation | 2.15 | same as main 2-bpp, but `H8` |
| old K=256 ablation | 2.15 | `d=8, K=256, M=2, H16, full-XTX recode=1` |

Do not repeat the full Hadamard block sweep or `d=16` sweep.
