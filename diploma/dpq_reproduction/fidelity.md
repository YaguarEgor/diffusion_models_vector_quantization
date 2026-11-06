# Fidelity Boundaries

Public packaging note: private infrastructure runners and scheduler receipts are
not included in this edition. Their generic activation-checkpoint wrapper is
extracted unchanged into `checkpointing.py`; the corresponding test imports that
module without changing its assertions. The quantization kernels and reference
searches are unchanged.

Author source: https://github.com/Jie-Shao-NJU/DPQ

Pinned commit: `12ee544b7dbaf48aecafad5c77c0a6b49be54e25`.

Paper: *Memory-Efficient Generative Models via Product Quantization*, ICCV2025,
https://cs.nju.edu.cn/wujx/paper/ICCV2025_DPQ.pdf.

The two arms must never be merged under a single unqualified "official DPQ" label.

| Component | `upstream` | `paper` |
|---|---|---|
| DiT and diffusion modules | untouched author source | same modules |
| PQ layout | untouched `CompressedLinear`, a book per input subspace | same product layout, compact shared pool + projection |
| Initialization | author config: SRC, requested100, internally forced1 | independent grouped Lloyd K-means20 |
| Codebook compression | absent | reconstructed frequency/radius pool |
| E-step | absent in ordinary training | subspace activation MSE, exact within-subspace XTX |
| M-step | all trainable parameters, as released | pool values only; other parameters frozen |
| Training schedule | respaced250, as released | original1000 forward noise schedule |
| DDPM objective | existing `training_losses`, epsilon MSE + learned-variance VB | same standard objective on the original schedule |
| Dead-code maintenance | absent | reservoir, ages, threshold256, trigger above3%, reinitialization |

## Explicit Reconstruction Choices

These details are not uniquely recoverable from the public implementation/text.
They are assumptions, not newly discovered author hyperparameters:

1. The pool target is `floor(O*I/(16*d*d))`, clamped to `[1,65536]` and the
   original centroid count for an unsigned16-bit projection. Threshold is .05 on
   raw Euclidean distance. The greedy pass uses descending assignment frequency,
   stable original order for ties. No additional normalization is inserted.
2. Projection chooses the closest usage frequency within that radius, then
   smallest distance/index. With no admissible entry after pool capacity is
   reached, nearest Euclidean distance is used. Selection/projection fallbacks
   and actual capacities are recorded per matrix. The advertised one-quarter
   budget is not forced by deleting additional entries: actual bytes are counted.
3. E-step runs for every forward microbatch. Each layer sees the current student
   input; its reference is the original pretrained matrix. Independent subspace
   errors are minimized, not a global true DDPM loss or full cross-subspace XTX.
   Checkpoint recomputation does not repeat E-step. Tie-breaking uses argmin.
4. Effective batch64, adaptive microbatch32/16/8/4/2/1 with accumulation. Different
   selected microbatches affect E-step statistics; the selected value is recorded.
   The batch size is a reproduction choice; it is not asserted to be specified in
   the final paper.
5. Reservoir size is `min(number_of_vectors, max(4096,2*pool_size))`; 256 random
   original weight vectors refresh it per optimizer update. Ages count optimizer
   updates and usage means references through the current assignment/projection.
6. Algorithm1 is implemented with scikit-learn's seeded k-means++ initialization
   on live centroids plus reservoir. Local centroid semantics are remapped to the
   new pool by nearest old-vector projection. Optimizer moments of a replaced
   pool are reset. This remapping/reset policy is not specified by the authors.
7. Published code optimizes unquantized modules and biases as well. The independent
   reconstruction freezes them to isolate the described centroid M-step. This
   difference is intentional and visible in reports, not an unnoticed fair-control
   guarantee. No extra teacher loss is added to either branch.

## Execution Changes, Not Algorithm Changes

The author source is never patched. A new runner removes cluster-specific paths,
adds deterministic seeds, block activation checkpointing, gradient accumulation,
atomic final saves and explicit resume checkpoints. Initialization can resume by
matrix. Training resume replays the seeded dataloader and uses stateless step
seeds; no automatic job resubmission is performed.

The `bounded-search-v2-stateless-reservoir` implementation preserves the E-step
objective and its frequency, but batches groups/rows with a 64 MiB distance
temporary budget. Assignment-change counts stay on GPU. Integer usage histograms
replace sorting of all assigned pool IDs. Reservoir sampling remains uniform
without replacement, now using Python's seeded sample instead of a full CUDA
permutation. This changes the RNG stream, not the sampling distribution. The
implementation and per-update/per-layer reservoir seed policy are recorded in
the protocol. Old checkpoints are not silently resumed under this version;
the original source-hash checks remain enforced. Floating-point near-ties can
change with batching, especially with TF32; GPU checks report these separately.

Training enables TF32 like the released code, disables fused attention, and
requests deterministic PyTorch kernels. Stored deployment values are FP16;
sampling materializes them in FP32 with TF32 disabled. Reservoirs, ages and
original reference matrices are excluded only from **deployment** accounting,
not claimed to disappear from training checkpoints.

## Evaluation

Initial target: 2-bit nominal indices, d4/K256, all140 linears in the28 blocks.
Both branches use one complete ImageNet **train** epoch, not a small proxy subset.
The extracted dataset is pinned by Hugging Face revision/shard hashes and includes
all1,281,167 images and all1000 classes. Center crop256, random horizontal flip,
VAE posterior sample and scale .18215 follow the released preprocessing.

Generation uses DDPM250, CFG1.5 with unchanged upstream three-channel guidance,
10k balanced seeded labels, the same VAE and ADM evaluator/reference used in the
previous DiT study. The paper does not specify identical seeds/class ordering.
FID/sFID/IS/precision/recall are compared to the corresponding article rows,
with a newly measured FP16-stored original DiT control. Exact equality is neither
expected nor used as proof that an implementation is correct.
