# Variance-head loss weight in VQDM calibration

The controlled variable is lambda in MSE(epsilon_Q,epsilon_T) + lambda MSE(v_Q,v_T), where v is the original model's raw learned-variance output. This is codebook calibration against the full-precision network output. Architecture, variance handling, native initialization, data, batches and storage are unchanged.

| Variance weight | Status at the snapshot | Matched10k FID |
|---:|---|---:|
|0|Complete|7.847127|
|0.1|Complete|7.814155|
|0.5|Submitted; waiting for scheduling|--|
|1|Submitted; waiting for scheduling|--|
|2|Submitted; waiting for scheduling|--|

Every new arm uses500 Adam updates and automatically proceeds to10000-image DDPM50 evaluation and ADM metrics after successful training. Detailed parameters, source-artifact checksums and CPU loss/gradient verification are in results.json. The0.033FID difference between the two completed controls is descriptive for fixed runs, not evidence of a statistically robust benefit.

The new jobs were submitted on12 November2026 and had not started at this snapshot. All completed results must come from the same matched10k cohort; no5k/10k or sampler changes are mixed. Operational launch recipes and private environment details are not part of this public record.
