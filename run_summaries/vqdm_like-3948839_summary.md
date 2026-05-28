# VQDM baseline run 3948839

Source log: `vqdm_like-3948839.log`

| target bpp | M | score | LPIPS | CLIP i-i | CLIP t-i | n |
|---:|---:|---:|---:|---:|---:|---:|
| 4.15 | 4 | 0.957463 | 0.0268 | 0.9843 | 0.3384 | 30 |
| 3.15 | 3 | 0.924190 | 0.0520 | 0.9762 | 0.3386 | 30 |
| 2.15 | 2 | 0.882565 | 0.0754 | 0.9579 | 0.3401 | 30 |

Notes:
- Baseline run used VQDM-style AVQ/AQ codebook training without finetuning.
- Evaluation here used n=30, so compare cautiously against RVQ runs with n=96.
