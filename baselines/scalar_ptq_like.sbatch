#!/bin/bash
#SBATCH --job-name=scalar_ptq
#SBATCH --output=runs/scalar_ptq-%j.log
#SBATCH --error=runs/scalar_ptq-%j.err
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

if [[ -f "${SLURM_SUBMIT_DIR}/quantize-scalar-ptq.py" ]]; then
  export SCALAR_PTQ_SCRIPT="${SLURM_SUBMIT_DIR}/quantize-scalar-ptq.py"
else
  echo "quantize-scalar-ptq.py was not found in ${SLURM_SUBMIT_DIR}" >&2
  echo "Expected server layout: quantize-scalar-ptq.py and scalar_ptq_like.sbatch in the same launch directory." >&2
  exit 1
fi

if [[ -f "${SLURM_SUBMIT_DIR}/pickscore_filtered_prompts_train.csv" && -f "${SLURM_SUBMIT_DIR}/pickscore_filtered_prompts_val.csv" ]]; then
  export PICKSCORE_DATA_DIR="${SLURM_SUBMIT_DIR}"
else
  echo "PickScore CSV files were not found in ${SLURM_SUBMIT_DIR}" >&2
  echo "Expected them next to scalar_ptq_like.sbatch in the same launch directory." >&2
  exit 1
fi
export SCALAR_PTQ_PICKSCORE_DATA_DIR="${PICKSCORE_DATA_DIR}"

export SCALAR_PTQ_SELECT_IDLE_GPUS=1
export SCALAR_PTQ_REQUIRE_IDLE_GPUS=1
export SCALAR_PTQ_VALIDATE_PRESET_GPUS=1
export SCALAR_PTQ_IDLE_GPU_MAX_USED_MEM_MIB=2048
export SCALAR_PTQ_AUTO_MAX_DEVICES=4

export SCALAR_PTQ_OUTPUT_ROOT="${SLURM_SUBMIT_DIR}/runs"
export SCALAR_PTQ_RUN_NAME="scalar_ptq_${SLURM_JOB_ID:-manual}"

export SCALAR_PTQ_BRANCHES=weight_only
export SCALAR_PTQ_WEIGHT_BITS=4,3,2
export SCALAR_PTQ_ACT_BITS=8
export SCALAR_PTQ_ACT_SCHEME=dynamic_per_sample

export SCALAR_PTQ_TARGET_MODE="${SCALAR_PTQ_TARGET_MODE:-all_midblock}"
export SCALAR_PTQ_INCLUDE_CONV="${SCALAR_PTQ_INCLUDE_CONV:-1}"
export SCALAR_PTQ_STEPS=50
export SCALAR_PTQ_GUIDANCE=5.0
export SCALAR_PTQ_EVAL_PROMPTS=32
export SCALAR_PTQ_EVAL_SEEDS=1234,4321,999
export SCALAR_PTQ_FORWARD_METRICS=1
export SCALAR_PTQ_FORWARD_METRIC_PROMPTS=2
export SCALAR_PTQ_FORWARD_METRIC_TIMESTEPS=3
export SCALAR_PTQ_FORWARD_METRIC_SEED=20260515

export SCALAR_PTQ_ALLOW_TF32=1
export SCALAR_PTQ_GEN_BATCH=0
export SCALAR_PTQ_GEN_BATCH_MIN=1
export SCALAR_PTQ_GEN_BATCH_MAX=32
export SCALAR_PTQ_GPU_TARGET_MEM_FRACTION=0.78
export SCALAR_PTQ_GPU_RESERVE_GIB=4
export SCALAR_PTQ_GEN_ESTIMATED_GIB_PER_IMAGE=1.8

python -V
nvidia-smi || true
python -u "${SCALAR_PTQ_SCRIPT}"
