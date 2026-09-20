#!/bin/bash
#SBATCH --job-name=vanilla_vq
#SBATCH --output=runs/vanilla_vq-%j.log
#SBATCH --error=runs/vanilla_vq-%j.err
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

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "${SLURM_SUBMIT_DIR}/quantize-vanilla-vq.py" ]]; then
  export VANILLA_VQ_SCRIPT="${SLURM_SUBMIT_DIR}/quantize-vanilla-vq.py"
elif [[ -f "${SCRIPT_DIR}/quantize-vanilla-vq.py" ]]; then
  export VANILLA_VQ_SCRIPT="${SCRIPT_DIR}/quantize-vanilla-vq.py"
else
  echo "quantize-vanilla-vq.py was not found in ${SLURM_SUBMIT_DIR} or ${SCRIPT_DIR}" >&2
  echo "Expected server layout: quantize-vanilla-vq.py and vanilla_vq_like.sbatch in the same launch directory." >&2
  exit 1
fi

if [[ -f "${SLURM_SUBMIT_DIR}/pickscore_filtered_prompts_train.csv" && -f "${SLURM_SUBMIT_DIR}/pickscore_filtered_prompts_val.csv" ]]; then
  export PICKSCORE_DATA_DIR="${SLURM_SUBMIT_DIR}"
else
  echo "PickScore CSV files were not found in ${SLURM_SUBMIT_DIR}" >&2
  echo "Expected them next to vanilla_vq_like.sbatch in the same launch directory." >&2
  exit 1
fi
export VANILLA_VQ_PICKSCORE_DATA_DIR="${PICKSCORE_DATA_DIR}"

export VANILLA_VQ_SELECT_IDLE_GPUS=1
export VANILLA_VQ_REQUIRE_IDLE_GPUS=1
export VANILLA_VQ_VALIDATE_PRESET_GPUS=1
export VANILLA_VQ_IDLE_GPU_MAX_USED_MEM_MIB=2048
export VANILLA_VQ_AUTO_MAX_DEVICES=4

export VANILLA_VQ_OUTPUT_ROOT="${SLURM_SUBMIT_DIR}/runs"
export VANILLA_VQ_RUN_NAME="vanilla_vq_${SLURM_JOB_ID:-manual}"

export VANILLA_VQ_BRANCHES=weight_only
export VANILLA_VQ_PROFILE_SPECS=bpp4_k256_d2,bpp3_k512_d3,bpp2_k256_d4,bpp1p5_k64_d4,bpp1_k256_d8
export VANILLA_VQ_CODEBOOK_BITS=16
export VANILLA_VQ_BIAS_BITS=16
export VANILLA_VQ_KMEANS_ITERS=18
export VANILLA_VQ_KMEANS_RESTARTS=1
export VANILLA_VQ_KMEANS_MAX_SAMPLES=131072
export VANILLA_VQ_ASSIGN_BATCH=65536

export VANILLA_VQ_TARGET_MODE="${VANILLA_VQ_TARGET_MODE:-all_midblock}"
export VANILLA_VQ_INCLUDE_CONV="${VANILLA_VQ_INCLUDE_CONV:-1}"
export VANILLA_VQ_STEPS=50
export VANILLA_VQ_GUIDANCE=5.0
export VANILLA_VQ_EVAL_PROMPTS=32
export VANILLA_VQ_EVAL_SEEDS=1234,4321,999

export VANILLA_VQ_CALIB_PROMPTS=0
export VANILLA_VQ_CALIB_SEEDS=777
export VANILLA_VQ_CALIB_STEPS=50
export VANILLA_VQ_CALIB_MAX_ROWS_PER_LAYER=4096
export VANILLA_VQ_CALIBRATION_POLICY=timestep_bucket_stats
export VANILLA_VQ_RECON_OBJECTIVE=weight_reconstruction_vanilla_vq
export VANILLA_VQ_TIMESTEP_BUCKET_WEIGHTS=1.0,1.0,1.0

export VANILLA_VQ_FORWARD_METRICS=1
export VANILLA_VQ_FORWARD_METRIC_PROMPTS=2
export VANILLA_VQ_FORWARD_METRIC_TIMESTEPS=3
export VANILLA_VQ_FORWARD_METRIC_SEED=20260515

export VANILLA_VQ_ALLOW_TF32=1
export VANILLA_VQ_GEN_BATCH=0
export VANILLA_VQ_GEN_BATCH_MIN=1
export VANILLA_VQ_GEN_BATCH_MAX=32
export VANILLA_VQ_GPU_TARGET_MEM_FRACTION=0.78
export VANILLA_VQ_GPU_RESERVE_GIB=4
export VANILLA_VQ_GEN_ESTIMATED_GIB_PER_IMAGE=1.8

python -V
nvidia-smi || true
python -u "${VANILLA_VQ_SCRIPT}"
