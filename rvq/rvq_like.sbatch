#!/bin/bash
#SBATCH --job-name=rvq_like
#SBATCH --output=runs/rvq_like-%j.log
#SBATCH --error=runs/rvq_like-%j.err
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=20:00:00
#SBATCH --constraint="[type_a|type_b|type_c]"

set -euo pipefail

module load Python/Anaconda_v03.2023

source "$(conda info --base)/etc/profile.d/conda.sh"
conda deactivate || true
conda activate vec_quant
``
cd "${SLURM_SUBMIT_DIR}"
mkdir -p runs

export PYTHONNOUSERSITE=1
unset PYTHONPATH

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "${SCRIPT_DIR}/quantize-vector-optimized.py" ]]; then
  export RVQ_SCRIPT="${SCRIPT_DIR}/quantize-vector-optimized.py"
elif [[ -f "${SLURM_SUBMIT_DIR}/quantize-vector-optimized.py" ]]; then
  export RVQ_SCRIPT="${SLURM_SUBMIT_DIR}/quantize-vector-optimized.py"
else
  echo "quantize-vector-optimized.py was not found in ${SLURM_SUBMIT_DIR} or ${SCRIPT_DIR}" >&2
  echo "Expected it either next to rvq_like.sbatch or in the submit directory." >&2
  exit 1
fi

if [[ -f "${SLURM_SUBMIT_DIR}/pickscore_filtered_prompts_train.csv" && -f "${SLURM_SUBMIT_DIR}/pickscore_filtered_prompts_val.csv" ]]; then
  export PICKSCORE_DATA_DIR="${SLURM_SUBMIT_DIR}"
else
  echo "PickScore CSV files were not found in ${SLURM_SUBMIT_DIR}" >&2
  exit 1
fi
export RVQ_PICKSCORE_DATA_DIR="${PICKSCORE_DATA_DIR}"

export RVQ_HW_PROFILE="${RVQ_HW_PROFILE:-v100x4}"
# Full 18-layer mid_block run by default. For the VQDM-style 16-layer scope use:
#   sbatch --export=ALL,RVQ_TARGET_MODE=hardcoded_midblock,RVQ_INCLUDE_CONV=1 rvq_like.sbatch
#
# Final RVQ candidate sweep: run four separate jobs to compare transform modes:
#   sbatch --export=ALL,RVQ_HADAMARD_MODE=noh rvq_like.sbatch
#   sbatch --export=ALL,RVQ_HADAMARD_MODE=h8  rvq_like.sbatch
#   sbatch --export=ALL,RVQ_HADAMARD_MODE=h16 rvq_like.sbatch
#   sbatch --export=ALL,RVQ_HADAMARD_MODE=h32 rvq_like.sbatch
export RVQ_FUNNEL_SWEEP="${RVQ_FUNNEL_SWEEP:-0}"
export RVQ_STAGE_K_SWEEP="${RVQ_STAGE_K_SWEEP:-1}"
export RVQ_HADAMARD_MODE="${RVQ_HADAMARD_MODE:-noh}"
export RVQ_TARGET_MODE="${RVQ_TARGET_MODE:-all_midblock}"
export RVQ_INCLUDE_CONV="${RVQ_INCLUDE_CONV:-1}"
export RVQ_FULL_XTX_RECODE_PASSES="${RVQ_FULL_XTX_RECODE_PASSES:-1}"
RVQ_RUN_STEM="rvq_fixed_${RVQ_TARGET_MODE}_${RVQ_HADAMARD_MODE}_${SLURM_JOB_ID:-manual}"
export RVQ_FIXED_SWEEP_OUTPUT_DIR="${RVQ_FIXED_SWEEP_OUTPUT_DIR:-${SLURM_SUBMIT_DIR}/runs/${RVQ_RUN_STEM}}"
export RVQ_FUNNEL_RESULTS_JSON="${RVQ_FUNNEL_RESULTS_JSON:-${SLURM_SUBMIT_DIR}/runs/rvq_funnel_${RVQ_TARGET_MODE}_${RVQ_HADAMARD_MODE}_${SLURM_JOB_ID:-manual}.json}"
export RVQ_COMPUTE_HPS=0
export RVQ_FORWARD_METRIC_PROMPTS=2
export RVQ_FORWARD_METRIC_TIMESTEPS=3
export RVQ_FORWARD_METRIC_SEED=20260515
mkdir -p "${RVQ_FIXED_SWEEP_OUTPUT_DIR}"

case "${RVQ_HADAMARD_MODE}" in
  noh|h8|h16|h32) ;;
  *)
    echo "Unsupported RVQ_HADAMARD_MODE=${RVQ_HADAMARD_MODE}; expected noh, h8, h16, or h32" >&2
    exit 1
    ;;
esac

echo "RVQ_SCRIPT=${RVQ_SCRIPT}"
echo "RVQ_TARGET_MODE=${RVQ_TARGET_MODE}"
echo "RVQ_INCLUDE_CONV=${RVQ_INCLUDE_CONV}"
echo "RVQ_STAGE_K_SWEEP=${RVQ_STAGE_K_SWEEP}"
echo "RVQ_HADAMARD_MODE=${RVQ_HADAMARD_MODE}"
echo "RVQ_FIXED_SWEEP_OUTPUT_DIR=${RVQ_FIXED_SWEEP_OUTPUT_DIR}"
echo "RVQ_FUNNEL_RESULTS_JSON=${RVQ_FUNNEL_RESULTS_JSON}"

python -V
python -u "${RVQ_SCRIPT}"
