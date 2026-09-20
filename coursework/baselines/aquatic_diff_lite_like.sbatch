#!/bin/bash
#SBATCH --job-name=aquatic_diff_lite
#SBATCH --output=runs/aquatic_diff_lite-%j.log
#SBATCH --error=runs/aquatic_diff_lite-%j.err
#SBATCH --gpus=4
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --constraint="[type_a|type_b|type_c]"

set -euo pipefail

module load Python/Anaconda_v03.2023

source "$(conda info --base)/etc/profile.d/conda.sh"
conda deactivate || true
conda activate vec_quant

cd "${SLURM_SUBMIT_DIR}"
mkdir -p runs

export PYTHONNOUSERSITE=1
unset PYTHONPATH

if [[ -f "${SLURM_SUBMIT_DIR}/quantize-aquatic-diff-lite.py" ]]; then
  export AQUATIC_SCRIPT="${SLURM_SUBMIT_DIR}/quantize-aquatic-diff-lite.py"
else
  echo "quantize-aquatic-diff-lite.py was not found in ${SLURM_SUBMIT_DIR}" >&2
  echo "Expected server layout: quantize-aquatic-diff-lite.py and aquatic_diff_lite_like.sbatch in the same launch directory." >&2
  exit 1
fi

if [[ -f "${SLURM_SUBMIT_DIR}/pickscore_filtered_prompts_train.csv" && -f "${SLURM_SUBMIT_DIR}/pickscore_filtered_prompts_val.csv" ]]; then
  export PICKSCORE_DATA_DIR="${SLURM_SUBMIT_DIR}"
else
  echo "PickScore CSV files were not found in ${SLURM_SUBMIT_DIR}" >&2
  echo "Expected them next to aquatic_diff_lite_like.sbatch in the same launch directory." >&2
  exit 1
fi
export AQUATIC_PICKSCORE_DATA_DIR="${PICKSCORE_DATA_DIR}"

export AQUATIC_SELECT_IDLE_GPUS=1
export AQUATIC_REQUIRE_IDLE_GPUS=1
export AQUATIC_VALIDATE_PRESET_GPUS=1
export AQUATIC_IDLE_GPU_MAX_USED_MEM_MIB=2048
export AQUATIC_AUTO_MAX_DEVICES=4

export AQUATIC_OUTPUT_ROOT="${SLURM_SUBMIT_DIR}/runs"
export AQUATIC_RUN_NAME="aquatic_diff_lite_${SLURM_JOB_ID:-manual}"

export AQUATIC_BRANCHES=weight_only
export AQUATIC_PROFILE_SPECS=bpp4_m4_k256_d8,bpp3_m3_k256_d8,bpp2_m2_k256_d8,bpp1p5_m2_k64_d8
export AQUATIC_CODEBOOK_BITS=16
export AQUATIC_BIAS_BITS=16
export AQUATIC_KMEANS_ITERS=18
export AQUATIC_KMEANS_RESTARTS=1
export AQUATIC_KMEANS_MAX_SAMPLES=131072
export AQUATIC_ASSIGN_BATCH=65536
export AQUATIC_REFINE_PASSES=1

export AQUATIC_TARGET_MODE="${AQUATIC_TARGET_MODE:-all_midblock}"
export AQUATIC_INCLUDE_CONV="${AQUATIC_INCLUDE_CONV:-1}"
export AQUATIC_STEPS=50
export AQUATIC_GUIDANCE=5.0
export AQUATIC_EVAL_PROMPTS=32
export AQUATIC_EVAL_SEEDS=1234,4321,999

export AQUATIC_CALIB_PROMPTS=16
export AQUATIC_CALIB_SEEDS=777
export AQUATIC_CALIB_STEPS=50
export AQUATIC_CALIB_MAX_ROWS_PER_LAYER=4096
export AQUATIC_CALIBRATION_POLICY=timestep_bucket_stats
export AQUATIC_RECON_OBJECTIVE=weight_reconstruction_additive_codebook
export AQUATIC_TIMESTEP_BUCKET_WEIGHTS=1.0,1.0,1.0

export AQUATIC_FORWARD_METRICS=1
export AQUATIC_FORWARD_METRIC_PROMPTS=2
export AQUATIC_FORWARD_METRIC_TIMESTEPS=3
export AQUATIC_FORWARD_METRIC_SEED=20260515

export AQUATIC_ALLOW_TF32=1
export AQUATIC_GEN_BATCH=0
export AQUATIC_GEN_BATCH_MIN=1
export AQUATIC_GEN_BATCH_MAX=32
export AQUATIC_GPU_TARGET_MEM_FRACTION=0.78
export AQUATIC_GPU_RESERVE_GIB=4
export AQUATIC_GEN_ESTIMATED_GIB_PER_IMAGE=1.8

python -V
nvidia-smi || true
python -u "${AQUATIC_SCRIPT}"
