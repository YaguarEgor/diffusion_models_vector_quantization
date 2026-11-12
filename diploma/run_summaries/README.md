# DiT Scientific Result Tables

Latest matched10k KD500 snapshot: [results and protocol](vqdm_kd500_matched_10k_20261112/).
Article/practicum artifacts: [my_first_article](https://github.com/YaguarEgor/my_first_article).


The tables preserve measured values, calibration/evaluation settings and uncertainty.
Operational identifiers, machine names and server locations are omitted in this public edition.
Column labels identifying experiment families remain meaningful; numeric metrics are unchanged.

| Directory | Contents |
|---|---|
| [dit_completed_20261016](dit_completed_20261016/) | Initial image metrics; temporal, structural and timestep diagnostics |
| [dit_followup_20261025](dit_followup_20261025/) | Local refits, paired comparisons and temporal-codebook geometry |
| [dit_objective_20261103](dit_objective_20261103/) | Temporal codebooks, objective comparisons and completed newer quality measurements |

[Scientific interpretation](../../docs/EXPERIMENTS.md) contains the assumptions and caveats.
These are dates of saved scientific records, not live scheduler status.

## Important Comparisons

- Initial FP/VQDM: common 10,000-image DDPM50 protocol.
- Temporal codebooks: matched image protocols and stated update budgets.
- Three-block teacher pilot: paired final latent NMSE, not FID.
- Fresh initialization: held-out teacher-output MSE, not completed VQDM initialization.
- New full-codebook variants: common 5,000-image protocol.
- `training_curves.csv`: six equally sized held-out timestep groups averaged at each saved update.
- `training_time.csv`: training-only timings, not a controlled hardware speed comparison.

Do not turn a 10,000-image baseline and a 5,000-image candidate into a claimed FID gain.

