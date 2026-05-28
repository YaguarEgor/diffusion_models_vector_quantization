# BitsFusion-lite Run, 2026-05-18

Metric convention matches the previous baseline summary:
`score = CLIP image-image - LPIPS`; higher is better. All rows were evaluated on
96 PickScore validation items: 32 prompts x 3 seeds, SD1.5, hardcoded 10 linear
mid-block layers.

## Run Status

- BitsFusion-lite job `4001347` completed normally.
- DGQ-lite was intentionally removed from the main baseline summaries because
  the implemented file was only a DGQ-inspired group/outlier PTQ variant, not a
  faithful DGQ reproduction with attention-score/alignment quantization.

## Headline

| Target | bpp | Score | CLIP i-i | LPIPS | Previous best non-VQDM | Delta |
| --- | ---: | ---: | ---: | ---: | --- | ---: |
| 4.15 | 4.136 | 0.7723 | 0.8981 | 0.1258 | PTQ4DM-lite 0.7629 | +0.0094 |
| 3.15 | 3.126 | 0.6301 | 0.8376 | 0.2074 | QDiffusion-lite 0.6097 | +0.0205 |
| 2.15 | 2.136 | 0.5224 | 0.7953 | 0.2729 | PTQ4DM-lite 0.4134 | +0.1090 |

## Interpretation

BitsFusion-lite is a useful stronger scalar/mixed-bit baseline. It improves the
non-VQDM baseline frontier at 3.15 and especially 2.15 bpp. The 2.15 bpp
allocation keeps two layers at 4 bits and puts eight layers at 2 bits, which is
much better than the all-2-bit scalar/PTQ4DM point.

Against RVQ, BitsFusion-lite still loses: by 0.0287 at 4.15, 0.0819 at 3.15,
and 0.0683 at 2.15 under strict RVQ bpp matching. Against VQDM it is also below
by 0.0714, 0.1239, and 0.1191 respectively.

## Files

- `bitsfusion_results.csv`: BitsFusion-lite rows.
- `comparison_vs_existing.csv`: BitsFusion-lite compared with the previous
  baseline frontier, strict RVQ, profile-best RVQ, and VQDM.
