# Run Summaries Index

This directory keeps compact experiment summaries only: CSV tables, Markdown notes, and small LaTeX tables. Raw logs, partial-only summaries, old duplicate reruns, and source configs are intentionally excluded.

## Final Baseline Summaries

| path | contents |
|---|---|
| `baselines_20260523_full_midblock/` | Full-midblock scalar baseline batch: Scalar PTQ, Q-Diffusion-lite, PTQ4DM-lite, plus older vector rows kept as historical context. This is still needed because the scalar methods are not fully replaced by later vector-only baseline batches. |
| `baselines_20260524_cluster_completed/` | Completed 18-layer cluster baseline rows, including VQDM, BitsFusion-lite, Vanilla VQ, and AQUATIC-Diff-like. Partial rows and older duplicate reruns are excluded. |
| `baselines_20260525_completed/` | Completed 16-layer vector baseline rows and AQUATIC-Diff 16/18-layer summaries. Partial RVQ rows from the same upload are excluded. |

Removed obsolete baseline summaries:

- `baselines_20260517/`: old 10-linear/partial baseline batch, replaced by later full-midblock summaries.
- `baselines_20260518_bitsfusion/`: old 10-linear BitsFusion-lite batch, replaced by later full-midblock summaries.
- `vqdm_like-3948839_*`: old n=30 VQDM run, replaced by completed VQDM summaries in the baseline folders.
- `vanilla_vq_4035902_full_summary.csv`: older duplicate of the completed Vanilla VQ full-midblock run; `4035907` is kept.

## RVQ Summaries

| path | contents |
|---|---|
| `all_completed_rvq_funnel_results.csv` | Normalized early RVQ ablation rows from jobs `3970494` and `3988130`; useful for d=8/d=16, fixed-`K`, Hadamard, and recode-history checks. |
| `rvq_experiment_results_summary.md` | Human-readable notes for early RVQ job `3970494`, including d=8 vs d=16 and full-XTX recode context. |
| `rvq_d8_d16_comparison.tex` | LaTeX table for the d=8 vs d=16 comparison. |
| `rvq_like-3988130_summary.md` | Summary of the later fixed-`K` RVQ run used for K/Hadamard/recode discussion. |
| `rvq_ablation_digest_20260524.md` | Compact digest of older RVQ hypotheses and conclusions. |
| `rvq_jobb_ablation_20260513_14/` | Consolidated JobB ablations: fixed-`K` recoding, stage-wise `K`, entropy/utilization, and local recode. |
| `rvq_final_20260531/` | Final completed RVQ stage-wise-`K` sweep. This is the main source for final RVQ headline tables. |
| `rvq_hadamard_clean_20260602/` | Isolated clean Hadamard-block sweep: fixed `K=256`, `d=8`, no recode, no stage-wise `K`, 39/39 variants complete. |

Removed obsolete or partial RVQ summaries:

- `rvq_cluster_20260527_status/`: interrupted partial rows only.
- `rvq_cluster_20260528_recovered/`: completed rows were incorporated into `rvq_final_20260531`; interrupted rows were not kept as final data.
- `rvq_cluster_20260530_status/`: interrupted partial rows only.
- `rvq_completed_results.csv` and `rvq_like-3988130_results.csv`: duplicated by `all_completed_rvq_funnel_results.csv`.

## Current Research Conclusions

1. VQDM remains the strongest baseline, but it is computationally expensive because of AVQ-style multi-codebook search.
2. The final RVQ variant is activation-aware RVQ with `d=8`, stage-wise `K`, full-XTX post-RVQ recode, and no local recode.
3. `d=16` did not help in completed paired comparisons and was roughly twice as slow.
4. Hadamard is useful only as a profile-dependent hyperparameter. The clean fixed-`K` sweep shows improvement mainly around the `3.15` profile.
5. Stage-wise `K` is one of the strongest improvements under a fixed bit budget.
6. Local recode after intermediate RVQ stages is not useful in the current implementation.
7. Final tables should use `rvq_final_20260531/` for RVQ and the completed baseline folders above for baselines; interrupted status folders have been removed.
