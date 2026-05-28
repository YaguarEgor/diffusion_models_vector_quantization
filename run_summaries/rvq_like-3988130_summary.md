# RVQ run 3988130 summary

Source logs:
- `runs/rvq_like-3988130.log`
- `runs/rvq_like-3988130.err`

Structured table:
- `run_summaries/rvq_like-3988130_results.csv`

## Status

The run was cancelled by Slurm, not by a Python/CUDA error:

```text
JOB 3988130 ON cn-010 CANCELLED AT 2026-05-14T07:36:33
```

No `Traceback`, CUDA OOM, or runtime exception was found. The final funnel JSON was not produced, so the saved CSV was reconstructed from completed `FUNNEL RESULT` lines in the log.

Important runtime note: although `rvq_like.sbatch` requested 4 GPUs, the script saw only one V100:

```text
cuda_totals_gib=[31.7]
Collect XTX with xtx_batch_size_cap=8, devices=['cuda']
```

So this run effectively used 1x V100 32GB.

## Completed Results

| stage | target bpp | K | family | H block | index bpp | score | LPIPS | CLIP i-i |
|---|---:|---:|---|---:|---:|---:|---:|---:|
| A:K256-d-block | 4.15 | 256 | H | 8 | 4.00 | 0.787277 | 0.1258 | 0.9131 |
| A:K256-d-block | 3.15 | 256 | H | 8 | 3.00 | 0.712077 | 0.1712 | 0.8833 |
| A:K256-d-block | 2.15 | 256 | H | 8 | 2.00 | 0.562665 | 0.2595 | 0.8222 |
| A:K256-d-block | 4.15 | 256 | H | 16 | 4.00 | 0.785055 | 0.1254 | 0.9105 |
| A:K256-d-block | 3.15 | 256 | H | 16 | 3.00 | 0.705097 | 0.1693 | 0.8744 |
| A:K256-d-block | 2.15 | 256 | H | 16 | 2.00 | 0.590704 | 0.2413 | 0.8320 |
| B:noH | 4.15 | 256 | noH |  | 4.00 | 0.800970 | 0.1175 | 0.9185 |
| B:noH | 3.15 | 256 | noH |  | 3.00 | 0.710136 | 0.1680 | 0.8781 |
| B:noH | 2.15 | 256 | noH |  | 2.00 | 0.568847 | 0.2563 | 0.8251 |
| C:K-ablation-H | 3.15 | 128 | H | 8 | 3.50 | 0.710735 | 0.1708 | 0.8815 |
| C:K-ablation-H | 3.15 | 512 | H | 8 | 3.38 | 0.758835 | 0.1382 | 0.8971 |
| C:K-ablation-noH | 3.15 | 128 | noH |  | 3.50 | 0.729444 | 0.1558 | 0.8852 |
| C:K-ablation-noH | 3.15 | 512 | noH |  | 3.38 | 0.761580 | 0.1395 | 0.9011 |
| C:K-ablation-H | 2.15 | 128 | H | 16 | 1.75 | 0.553258 | 0.2653 | 0.8186 |
| C:K-ablation-H | 2.15 | 512 | H | 16 | 2.25 | 0.620219 | 0.2235 | 0.8437 |
| C:K-ablation-noH | 2.15 | 128 | noH |  | 1.75 | 0.552668 | 0.2659 | 0.8186 |
| C:K-ablation-noH | 2.15 | 512 | noH |  | 2.25 | 0.626237 | 0.2156 | 0.8419 |
| D:all-linear | 2.15 | 512 | H | 16 | 2.25 | 0.620219 | 0.2235 | 0.8437 |

## Best Completed Results

| target bpp | best completed variant | index bpp | score | LPIPS | CLIP i-i |
|---:|---|---:|---:|---:|---:|
| 4.15 | K=256 noH | 4.00 | 0.800970 | 0.1175 | 0.9185 |
| 3.15 | K=512 noH | 3.38 | 0.761580 | 0.1395 | 0.9011 |
| 2.15 | K=512 noH | 2.25 | 0.626237 | 0.2156 | 0.8419 |

## Notes

- Increasing K helped strongly, especially at low bpp.
- At 2.15 target, K=512 noH improved over K=256 H16: score 0.626237 vs 0.590704, LPIPS 0.2156 vs 0.2413.
- This is not a perfectly equal-bpp comparison: K=512 raises packed index bpp from 2.00 to 2.25 for M=2.
- Hadamard is mixed: it helped K=256 at 2.15, but noH was better for K=512 at both 2.15 and 3.15 in this run.
- The D all-linear result duplicated the C K=512 H16 result, suggesting that this all-linear target set matched the same 10 hardcoded Linear layers.
