#!/bin/bash
#SBATCH --job-name=qdiff_lite
#SBATCH --output=runs/qdiff_lite-%j.log
#SBATCH --error=runs/qdiff_lite-%j.err
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

if [[ -f "${SLURM_SUBMIT_DIR}/quantize-qdiffusion-lite.py" ]]; then
  export QDIFF_SCRIPT="${SLURM_SUBMIT_DIR}/quantize-qdiffusion-lite.py"
else
  echo "quantize-qdiffusion-lite.py was not found in ${SLURM_SUBMIT_DIR}" >&2
  echo "Expected server layout: quantize-qdiffusion-lite.py and qdiffusion_lite_like.sbatch in the same launch directory." >&2
  exit 1
fi

if [[ -f "${SLURM_SUBMIT_DIR}/pickscore_filtered_prompts_train.csv" && -f "${SLURM_SUBMIT_DIR}/pickscore_filtered_prompts_val.csv" ]]; then
  export PICKSCORE_DATA_DIR="${SLURM_SUBMIT_DIR}"
else
  echo "PickScore CSV files were not found in ${SLURM_SUBMIT_DIR}" >&2
  echo "Expected them next to qdiffusion_lite_like.sbatch in the same launch directory." >&2
  exit 1
fi
export QDIFF_PICKSCORE_DATA_DIR="${PICKSCORE_DATA_DIR}"

export QDIFF_SELECT_IDLE_GPUS=1
export QDIFF_REQUIRE_IDLE_GPUS=1
export QDIFF_VALIDATE_PRESET_GPUS=1
export QDIFF_IDLE_GPU_MAX_USED_MEM_MIB=2048
export QDIFF_AUTO_MAX_DEVICES=4

export QDIFF_OUTPUT_ROOT="${SLURM_SUBMIT_DIR}/runs"
export QDIFF_RUN_NAME="qdiffusion_lite_${SLURM_JOB_ID:-manual}"

export QDIFF_BRANCHES=weight_only
export QDIFF_WEIGHT_BITS=4,3,2
export QDIFF_ACT_BITS=8
export QDIFF_ACT_SCHEME=dynamic_per_sample

export QDIFF_TARGET_MODE="${QDIFF_TARGET_MODE:-all_midblock}"
export QDIFF_INCLUDE_CONV="${QDIFF_INCLUDE_CONV:-1}"
export QDIFF_STEPS=50
export QDIFF_GUIDANCE=5.0
export QDIFF_EVAL_PROMPTS=32
export QDIFF_EVAL_SEEDS=1234,4321,999

export QDIFF_CALIB_PROMPTS=16
export QDIFF_CALIB_SEEDS=777
export QDIFF_CALIB_STEPS=50
export QDIFF_CALIB_MAX_ROWS_PER_LAYER=4096
export QDIFF_SCALE_GRID=0.5,0.625,0.75,0.875,1.0,1.125,1.25,1.5,1.75,2.0
export QDIFF_SCALE_OPT_OUT_CHUNK=256

export QDIFF_FORWARD_METRICS=1
export QDIFF_FORWARD_METRIC_PROMPTS=2
export QDIFF_FORWARD_METRIC_TIMESTEPS=3
export QDIFF_FORWARD_METRIC_SEED=20260515

export QDIFF_ALLOW_TF32=1
export QDIFF_GEN_BATCH=0
export QDIFF_GEN_BATCH_MIN=1
export QDIFF_GEN_BATCH_MAX=32
export QDIFF_GPU_TARGET_MEM_FRACTION=0.78
export QDIFF_GPU_RESERVE_GIB=4
export QDIFF_GEN_ESTIMATED_GIB_PER_IMAGE=1.8

python -V
nvidia-smi || true
python -u "${QDIFF_SCRIPT}"
