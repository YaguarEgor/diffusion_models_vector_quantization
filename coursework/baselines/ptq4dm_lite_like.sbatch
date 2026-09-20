#!/bin/bash
#SBATCH --job-name=ptq4dm_lite
#SBATCH --output=runs/ptq4dm_lite-%j.log
#SBATCH --error=runs/ptq4dm_lite-%j.err
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

if [[ -f "${SLURM_SUBMIT_DIR}/quantize-ptq4dm-lite.py" ]]; then
  export PTQ4DM_SCRIPT="${SLURM_SUBMIT_DIR}/quantize-ptq4dm-lite.py"
else
  echo "quantize-ptq4dm-lite.py was not found in ${SLURM_SUBMIT_DIR}" >&2
  echo "Expected server layout: quantize-ptq4dm-lite.py and ptq4dm_lite_like.sbatch in the same launch directory." >&2
  exit 1
fi

if [[ -f "${SLURM_SUBMIT_DIR}/pickscore_filtered_prompts_train.csv" && -f "${SLURM_SUBMIT_DIR}/pickscore_filtered_prompts_val.csv" ]]; then
  export PICKSCORE_DATA_DIR="${SLURM_SUBMIT_DIR}"
else
  echo "PickScore CSV files were not found in ${SLURM_SUBMIT_DIR}" >&2
  echo "Expected them next to ptq4dm_lite_like.sbatch in the same launch directory." >&2
  exit 1
fi
export PTQ4DM_PICKSCORE_DATA_DIR="${PICKSCORE_DATA_DIR}"

export PTQ4DM_SELECT_IDLE_GPUS=1
export PTQ4DM_REQUIRE_IDLE_GPUS=1
export PTQ4DM_VALIDATE_PRESET_GPUS=1
export PTQ4DM_IDLE_GPU_MAX_USED_MEM_MIB=2048
export PTQ4DM_AUTO_MAX_DEVICES=4

export PTQ4DM_OUTPUT_ROOT="${SLURM_SUBMIT_DIR}/runs"
export PTQ4DM_RUN_NAME="ptq4dm_lite_${SLURM_JOB_ID:-manual}"

export PTQ4DM_BRANCHES=weight_only
export PTQ4DM_WEIGHT_BITS=4,3,2
export PTQ4DM_ACT_BITS=8
export PTQ4DM_ACT_SCHEME=dynamic_per_sample

export PTQ4DM_TARGET_MODE="${PTQ4DM_TARGET_MODE:-all_midblock}"
export PTQ4DM_INCLUDE_CONV="${PTQ4DM_INCLUDE_CONV:-1}"
export PTQ4DM_STEPS=50
export PTQ4DM_GUIDANCE=5.0
export PTQ4DM_EVAL_PROMPTS=32
export PTQ4DM_EVAL_SEEDS=1234,4321,999

export PTQ4DM_CALIB_PROMPTS=16
export PTQ4DM_CALIB_SEEDS=777
export PTQ4DM_CALIB_STEPS=50
export PTQ4DM_CALIB_MAX_ROWS_PER_LAYER=4096
export PTQ4DM_CALIBRATION_POLICY=layer_type+timestep_bucket
export PTQ4DM_RECON_OBJECTIVE=timestep_weighted_layer_output
export PTQ4DM_TIMESTEP_BUCKET_WEIGHTS=1.0,1.0,1.0
export PTQ4DM_CLIP_RATIO_GRID=0.4,0.5,0.6,0.7,0.8,0.9,0.95,1.0
export PTQ4DM_CLIP_RATIO_GRID_ATTN=0.5,0.6,0.7,0.8,0.9,0.95,1.0
export PTQ4DM_CLIP_RATIO_GRID_FFN=0.4,0.5,0.6,0.7,0.8,0.9,0.95,1.0
export PTQ4DM_CLIP_RATIO_GRID_CONV=0.6,0.7,0.8,0.9,0.95,1.0
export PTQ4DM_CLIP_OPT_OUT_CHUNK=256

export PTQ4DM_FORWARD_METRICS=1
export PTQ4DM_FORWARD_METRIC_PROMPTS=2
export PTQ4DM_FORWARD_METRIC_TIMESTEPS=3
export PTQ4DM_FORWARD_METRIC_SEED=20260515

export PTQ4DM_ALLOW_TF32=1
export PTQ4DM_GEN_BATCH=0
export PTQ4DM_GEN_BATCH_MIN=1
export PTQ4DM_GEN_BATCH_MAX=32
export PTQ4DM_GPU_TARGET_MEM_FRACTION=0.78
export PTQ4DM_GPU_RESERVE_GIB=4
export PTQ4DM_GEN_ESTIMATED_GIB_PER_IMAGE=1.8

python -V
nvidia-smi || true
python -u "${PTQ4DM_SCRIPT}"
