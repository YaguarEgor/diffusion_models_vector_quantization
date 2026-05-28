# Completed Cluster Runs Summary

Generated: 2026-05-24 22:36:59 local time.

Only runs whose own log contains `Run finished` are included here. Partial/cancelled runs were intentionally excluded from score tables.

## Included Runs

| Method | Job | Scope | Rows | Full CSV |
|---|---:|---|---:|---|
| VQDM | 4036922 | full midblock, 18 layers | 3 | `vqdm_4036922_full_summary.csv` |
| BitsFusion-lite | 4035831 | full midblock, 18 layers | 4 | `bitsfusion_lite_4035831_full_summary.csv` |
| Vanilla VQ | 4035902 | full midblock, 18 layers | 5 | `vanilla_vq_4035902_full_summary.csv` |
| Vanilla VQ | 4035907 | full midblock, 18 layers, duplicate seed/check run | 5 | `vanilla_vq_4035907_full_summary.csv` |
| AQUATIC-Diff-like | 4035857 | VQDM-style 16 layers, time_emb_proj skipped | 4 | `aquatic_diff_like_4035857_full_summary.csv` |

## Compact Metrics

| Method | Job | Layers | Target bpp | Actual bpp | Score | CLIP i-i | LPIPS | CLIP t-i | PSNR | SSIM | Quant s | Eval s | Total s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| AQUATIC-Diff-like | 4035857 | 16 | 4.000 | 4.0224 | 0.6731 | 0.8515 | 0.1784 | 0.3225 | 20.81 | 0.8801 | 5.1 | 222.4 | 228.1 |
| AQUATIC-Diff-like | 4035857 | 16 | 3.000 | 3.0168 | 0.5891 | 0.8205 | 0.2314 | 0.3232 | 19.01 | 0.8351 | 2.7 | 213.9 | 216.9 |
| AQUATIC-Diff-like | 4035857 | 16 | 2.000 | 2.0112 | 0.4635 | 0.7649 | 0.3014 | 0.3105 | 17.05 | 0.7707 | 1.8 | 213.5 | 215.6 |
| AQUATIC-Diff-like | 4035857 | 16 | 1.500 | 1.5028 | 0.3529 | 0.7151 | 0.3623 | 0.2969 | 16.28 | 0.7286 | 0.9 | 213.5 | 214.7 |
| BitsFusion-lite | 4035831 | 18 | 4.150 | 4.0053 | 0.7131 | 0.8709 | 0.1578 | 0.3210 | 21.34 | 0.8911 | 6.0 | 224.4 | 231.5 |
| BitsFusion-lite | 4035831 | 18 | 3.050 | 3.0424 | 0.6109 | 0.8289 | 0.2180 | 0.3193 | 19.19 | 0.8522 | 6.0 | 215.5 | 221.9 |
| BitsFusion-lite | 4035831 | 18 | 2.050 | 2.0458 | 0.4677 | 0.7678 | 0.3001 | 0.3121 | 17.32 | 0.7818 | 6.0 | 215.3 | 221.7 |
| BitsFusion-lite | 4035831 | 18 | 1.500 | 1.4951 | 0.3157 | 0.6969 | 0.3813 | 0.3003 | 16.03 | 0.7171 | 6.0 | 215.3 | 221.7 |
| VQDM | 4036922 | 18 | 4.150 | 4.0296 | 0.8253 | 0.9258 | 0.1005 | 0.3095 | 24.84 | 0.9318 | 3314.0 | 227.9 | 3542.3 |
| VQDM | 4036922 | 18 | 3.150 | 3.0235 | 0.7210 | 0.8852 | 0.1642 | 0.3085 | 21.68 | 0.8900 | 2846.8 | 226.0 | 3073.4 |
| VQDM | 4036922 | 18 | 2.150 | 2.0174 | 0.5865 | 0.8305 | 0.2440 | 0.3051 | 19.15 | 0.8294 | 1857.1 | 224.7 | 2082.0 |
| Vanilla VQ | 4035902 | 18 | 4.000 | 4.0015 | 0.7374 | 0.8814 | 0.1440 | 0.3170 | 21.86 | 0.9070 | 1.6 | 220.5 | 222.5 |
| Vanilla VQ | 4035907 | 18 | 4.000 | 4.0015 | 0.7374 | 0.8814 | 0.1440 | 0.3170 | 21.86 | 0.9070 | 1.5 | 220.5 | 222.4 |
| Vanilla VQ | 4035902 | 18 | 3.000 | 3.0053 | 0.6166 | 0.8283 | 0.2117 | 0.3195 | 19.49 | 0.8537 | 1.9 | 213.3 | 215.6 |
| Vanilla VQ | 4035907 | 18 | 3.000 | 3.0053 | 0.6134 | 0.8270 | 0.2135 | 0.3185 | 19.49 | 0.8535 | 2.0 | 213.5 | 215.9 |
| Vanilla VQ | 4035902 | 18 | 2.000 | 2.0030 | 0.4495 | 0.7634 | 0.3139 | 0.3113 | 16.92 | 0.7691 | 1.0 | 213.4 | 214.7 |
| Vanilla VQ | 4035907 | 18 | 2.000 | 2.0030 | 0.4441 | 0.7606 | 0.3165 | 0.3135 | 16.89 | 0.7671 | 1.0 | 213.4 | 214.7 |
| Vanilla VQ | 4035902 | 18 | 1.500 | 1.5008 | 0.3549 | 0.7182 | 0.3634 | 0.3030 | 16.15 | 0.7285 | 0.5 | 213.3 | 214.2 |
| Vanilla VQ | 4035907 | 18 | 1.500 | 1.5008 | 0.3531 | 0.7164 | 0.3633 | 0.3026 | 16.16 | 0.7293 | 0.5 | 213.6 | 214.5 |
| Vanilla VQ | 4035902 | 18 | 1.000 | 1.0061 | 0.3054 | 0.6947 | 0.3893 | 0.2970 | 15.78 | 0.7048 | 0.8 | 213.3 | 214.5 |
| Vanilla VQ | 4035907 | 18 | 1.000 | 1.0061 | 0.3075 | 0.6962 | 0.3887 | 0.2966 | 15.78 | 0.7033 | 0.9 | 213.4 | 214.6 |

## Excluded Partial Runs

| Run | Reason |
|---|---|
| `vqdm_full_midblock_4035853` | partial/cancelled; superseded by completed VQDM 4036922 |
| `vqdm_vqdmstyle16_4035855` | partial/cancelled; only first VQDM-style16 profile finished |
| `rvq_like-4035923` | partial/cancelled RVQ full 18; no final summary |
| `rvq_like-4035950` | partial/cancelled RVQ VQDM-style16; no final summary |

## Notes

- VQDM `4036922` is the completed full-midblock 18-layer reference and supersedes the partial VQDM `4035853`.
- Two Vanilla VQ runs are both complete full-midblock runs; keep both as repeat/check runs or average them later if needed.
- AQUATIC-Diff-like here is the VQDM-style 16-layer run, so compare it with 18-layer methods only with that scope caveat.
