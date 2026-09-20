#!/bin/bash
#SBATCH --job-name=bitsfusion_lite
#SBATCH --output=runs/bitsfusion_lite-%j.log
#SBATCH --error=runs/bitsfusion_lite-%j.err
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

if [[ -f "${SLURM_SUBMIT_DIR}/quantize-bitsfusion-lite.py" ]]; then
  export BITSFUSION_SCRIPT="${SLURM_SUBMIT_DIR}/quantize-bitsfusion-lite.py"
else
  echo "quantize-bitsfusion-lite.py was not found in ${SLURM_SUBMIT_DIR}" >&2
  echo "Expected server layout: quantize-bitsfusion-lite.py and bitsfusion_lite_like.sbatch in the same launch directory." >&2
  exit 1
fi

if [[ -f "${SLURM_SUBMIT_DIR}/pickscore_filtered_prompts_train.csv" && -f "${SLURM_SUBMIT_DIR}/pickscore_filtered_prompts_val.csv" ]]; then
  export PICKSCORE_DATA_DIR="${SLURM_SUBMIT_DIR}"
else
  echo "PickScore CSV files were not found in ${SLURM_SUBMIT_DIR}" >&2
  echo "Expected them next to bitsfusion_lite_like.sbatch in the same launch directory." >&2
  exit 1
fi
export BITSFUSION_PICKSCORE_DATA_DIR="${PICKSCORE_DATA_DIR}"

export BITSFUSION_SELECT_IDLE_GPUS=1
export BITSFUSION_REQUIRE_IDLE_GPUS=1
export BITSFUSION_VALIDATE_PRESET_GPUS=1
export BITSFUSION_IDLE_GPU_MAX_USED_MEM_MIB=2048
export BITSFUSION_AUTO_MAX_DEVICES=4

export BITSFUSION_OUTPUT_ROOT="${SLURM_SUBMIT_DIR}/runs"
export BITSFUSION_RUN_NAME="bitsfusion_lite_${SLURM_JOB_ID:-manual}"

export BITSFUSION_TARGET_BPPS=4.15,3.05,2.05,1.50
export BITSFUSION_LAYER_BITS=1,2,3,4
export BITSFUSION_ACT_BITS=0
export BITSFUSION_ACT_SCHEME=dynamic_per_sample

export BITSFUSION_TARGET_MODE="${BITSFUSION_TARGET_MODE:-all_midblock}"
export BITSFUSION_INCLUDE_CONV="${BITSFUSION_INCLUDE_CONV:-1}"
export BITSFUSION_STEPS=50
export BITSFUSION_GUIDANCE=5.0
export BITSFUSION_EVAL_PROMPTS=32
export BITSFUSION_EVAL_SEEDS=1234,4321,999

export BITSFUSION_CALIB_PROMPTS=16
export BITSFUSION_CALIB_SEEDS=777
export BITSFUSION_CALIB_STEPS=50
export BITSFUSION_CALIB_MAX_ROWS_PER_LAYER=4096
export BITSFUSION_ALLOCATION_POLICY=semantic_sensitivity_greedy
export BITSFUSION_BALANCED_INTEGER=1
export BITSFUSION_SEMANTIC_PROTECT=1
export BITSFUSION_SEMANTIC_MIN_BITS=2
export BITSFUSION_CROSS_ATTN_SENSITIVITY=1.50
export BITSFUSION_SELF_ATTN_SENSITIVITY=1.25
export BITSFUSION_FFN_SENSITIVITY=1.00
export BITSFUSION_CONV_SENSITIVITY=1.10
export BITSFUSION_TIMESTEP_BUCKET_WEIGHTS=1.0,1.0,1.0
export BITSFUSION_CLIP_RATIO_GRID=0.4,0.5,0.6,0.7,0.8,0.9,0.95,1.0
export BITSFUSION_CLIP_OPT_OUT_CHUNK=256

export BITSFUSION_FORWARD_METRICS=1
export BITSFUSION_FORWARD_METRIC_PROMPTS=2
export BITSFUSION_FORWARD_METRIC_TIMESTEPS=3
export BITSFUSION_FORWARD_METRIC_SEED=20260515

export BITSFUSION_ALLOW_TF32=1
export BITSFUSION_GEN_BATCH=0
export BITSFUSION_GEN_BATCH_MIN=1
export BITSFUSION_GEN_BATCH_MAX=32
export BITSFUSION_GPU_TARGET_MEM_FRACTION=0.78
export BITSFUSION_GPU_RESERVE_GIB=4
export BITSFUSION_GEN_ESTIMATED_GIB_PER_IMAGE=1.8

python -V
nvidia-smi || true
python -u "${BITSFUSION_SCRIPT}"
