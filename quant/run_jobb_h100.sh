#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
BUNDLE_DIR="${BUNDLE_DIR:-runs/jobb_terminal_${RUN_ID}}"
RESULTS_DIR="${BUNDLE_DIR}/results"
LOG_DIR="${BUNDLE_DIR}/logs"
ENV_DIR="${BUNDLE_DIR}/environment"
SOURCE_DIR="${BUNDLE_DIR}/source"
ARCHIVE_PATH="${BUNDLE_DIR}.tar.gz"
VENV_DIR="${VENV_DIR:-${ROOT_DIR}/.venv_jobb}"

mkdir -p "${RESULTS_DIR}" "${LOG_DIR}" "${ENV_DIR}" "${SOURCE_DIR}" runs

main() {
  echo "JobB run id: ${RUN_ID}"
  echo "Root dir: ${ROOT_DIR}"
  echo "Bundle dir: ${BUNDLE_DIR}"
  echo "Results dir: ${RESULTS_DIR}"
  date -Is | tee "${BUNDLE_DIR}/started_at.txt"

  if [[ ! -f quantize-vector-jobb.py ]]; then
    echo "ERROR: quantize-vector-jobb.py is missing in ${ROOT_DIR}" >&2
    exit 2
  fi
  if [[ ! -f requirements-jobb.txt ]]; then
    echo "ERROR: requirements-jobb.txt is missing in ${ROOT_DIR}" >&2
    exit 4
  fi

  cp quantize-vector-jobb.py "${SOURCE_DIR}/"
  [[ -f requirements-jobb.txt ]] && cp requirements-jobb.txt "${SOURCE_DIR}/"
  [[ -f requirements.txt ]] && cp requirements.txt "${SOURCE_DIR}/"
  [[ -f pickscore_filtered_prompts_train.csv ]] && cp pickscore_filtered_prompts_train.csv "${SOURCE_DIR}/"
  [[ -f pickscore_filtered_prompts_val.csv ]] && cp pickscore_filtered_prompts_val.csv "${SOURCE_DIR}/"

  export PYTHONNOUSERSITE=1
  export PYTHONHASHSEED="${PYTHONHASHSEED:-20260511}"
  export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
  unset PYTHONPATH

  export PICKSCORE_DATA_DIR="${PICKSCORE_DATA_DIR:-${ROOT_DIR}}"
  export RVQ_PICKSCORE_DATA_DIR="${RVQ_PICKSCORE_DATA_DIR:-${ROOT_DIR}}"
  export RVQ_HW_PROFILE="${RVQ_HW_PROFILE:-h100}"

  select_idle_gpus

  export RVQ_AUTO_MAX_DEVICES="${RVQ_AUTO_MAX_DEVICES:-${SELECTED_IDLE_GPU_COUNT:-1}}"
  export RVQ_GPU_AUTOTUNE="${RVQ_GPU_AUTOTUNE:-1}"
  export RVQ_USE_MULTI_GPU_ASSIGN="${RVQ_USE_MULTI_GPU_ASSIGN:-1}"
  export RVQ_USE_MULTI_GPU_XTX="${RVQ_USE_MULTI_GPU_XTX:-1}"
  export RVQ_EVAL_INPLACE_MIDBLOCK="${RVQ_EVAL_INPLACE_MIDBLOCK:-1}"
  export RVQ_COMPUTE_HPS="${RVQ_COMPUTE_HPS:-0}"
  export RVQ_ALLOW_TF32="${RVQ_ALLOW_TF32:-1}"
  export RVQ_AUTO_INSTALL_DEPS="${RVQ_AUTO_INSTALL_DEPS:-0}"
  export RVQ_JOBB_RUN_ID="${RUN_ID}"
  export RVQ_JOBB_OUTPUT_DIR="${RESULTS_DIR}"
  export RVQ_JOBB_RECODE_PASSES="${RVQ_JOBB_RECODE_PASSES:-1}"
  export RVQ_JOBB_LOCAL_RECODE_PASSES="${RVQ_JOBB_LOCAL_RECODE_PASSES:-0,1}"
  export RVQ_JOBB_ALL_LINEAR_TOPK_PER_GROUP="${RVQ_JOBB_ALL_LINEAR_TOPK_PER_GROUP:-1}"
  export RVQ_JOBB_CONV_TOPK_PER_GROUP="${RVQ_JOBB_CONV_TOPK_PER_GROUP:-0}"

  for csv_file in pickscore_filtered_prompts_train.csv pickscore_filtered_prompts_val.csv; do
    if [[ ! -f "${RVQ_PICKSCORE_DATA_DIR}/${csv_file}" && ! -f "${PICKSCORE_DATA_DIR}/${csv_file}" ]]; then
      echo "ERROR: ${csv_file} not found in RVQ_PICKSCORE_DATA_DIR=${RVQ_PICKSCORE_DATA_DIR} or PICKSCORE_DATA_DIR=${PICKSCORE_DATA_DIR}" >&2
      exit 3
    fi
  done

  {
    echo "RUN_ID=${RUN_ID}"
    echo "BUNDLE_DIR=${BUNDLE_DIR}"
    echo "VENV_DIR=${VENV_DIR}"
    echo "RVQ_JOBB_OUTPUT_DIR=${RVQ_JOBB_OUTPUT_DIR}"
    echo "PICKSCORE_DATA_DIR=${PICKSCORE_DATA_DIR}"
    echo "RVQ_PICKSCORE_DATA_DIR=${RVQ_PICKSCORE_DATA_DIR}"
    echo "RVQ_HW_PROFILE=${RVQ_HW_PROFILE}"
    echo "RVQ_AUTO_MAX_DEVICES=${RVQ_AUTO_MAX_DEVICES}"
    echo "RVQ_IDLE_GPU_MAX_USED_MEM_MIB=${RVQ_IDLE_GPU_MAX_USED_MEM_MIB:-2048}"
    echo "RVQ_GPU_AUTOTUNE=${RVQ_GPU_AUTOTUNE}"
    echo "RVQ_USE_MULTI_GPU_ASSIGN=${RVQ_USE_MULTI_GPU_ASSIGN}"
    echo "RVQ_USE_MULTI_GPU_XTX=${RVQ_USE_MULTI_GPU_XTX}"
    echo "RVQ_EVAL_INPLACE_MIDBLOCK=${RVQ_EVAL_INPLACE_MIDBLOCK}"
    echo "RVQ_COMPUTE_HPS=${RVQ_COMPUTE_HPS}"
    echo "RVQ_ALLOW_TF32=${RVQ_ALLOW_TF32}"
    echo "RVQ_AUTO_INSTALL_DEPS=${RVQ_AUTO_INSTALL_DEPS}"
    echo "RVQ_JOBB_RECODE_PASSES=${RVQ_JOBB_RECODE_PASSES}"
    echo "RVQ_JOBB_LOCAL_RECODE_PASSES=${RVQ_JOBB_LOCAL_RECODE_PASSES}"
    echo "RVQ_JOBB_ALL_LINEAR_TOPK_PER_GROUP=${RVQ_JOBB_ALL_LINEAR_TOPK_PER_GROUP}"
    echo "RVQ_JOBB_CONV_TOPK_PER_GROUP=${RVQ_JOBB_CONV_TOPK_PER_GROUP}"
    echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
    echo "PYTHONHASHSEED=${PYTHONHASHSEED}"
    echo "CUBLAS_WORKSPACE_CONFIG=${CUBLAS_WORKSPACE_CONFIG}"
  } > "${ENV_DIR}/run_config.env"

  echo
  echo "GPU snapshot:"
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi | tee "${ENV_DIR}/nvidia-smi-start.txt"
    nvidia-smi -L | tee "${ENV_DIR}/nvidia-smi-list.txt"
  else
    echo "nvidia-smi not found" | tee "${ENV_DIR}/nvidia-smi-start.txt"
  fi

  echo
  echo "Creating/updating Python virtual environment: ${VENV_DIR}"
  if [[ -z "${BASE_PYTHON:-}" ]]; then
    for candidate in python3.12 python3.11 python3.10 python3 python; do
      if ! command -v "${candidate}" >/dev/null 2>&1; then
        continue
      fi
      candidate_supported="$(${candidate} -c 'import sys; print("1" if (sys.version_info.major, sys.version_info.minor) >= (3, 10) and (sys.version_info.major, sys.version_info.minor) <= (3, 12) else "0")')"
      if [[ "${candidate_supported}" == "1" ]]; then
        BASE_PYTHON="${candidate}"
        break
      fi
      echo "Skipping ${candidate}: torch==2.4.1+cu118 pins require Python 3.10-3.12"
    done
  fi
  if [[ -z "${BASE_PYTHON:-}" ]]; then
    echo "ERROR: no suitable Python 3.10-3.12 interpreter found for torch==2.4.1+cu118" >&2
    exit 5
  fi
  BASE_PYTHON_VERSION="$(${BASE_PYTHON} -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
  BASE_PYTHON_SUPPORTED="$(${BASE_PYTHON} -c 'import sys; print("1" if (sys.version_info.major, sys.version_info.minor) >= (3, 10) and (sys.version_info.major, sys.version_info.minor) <= (3, 12) else "0")')"
  if [[ "${BASE_PYTHON_SUPPORTED}" != "1" ]]; then
    echo "ERROR: BASE_PYTHON=${BASE_PYTHON} resolves to Python ${BASE_PYTHON_VERSION}; torch==2.4.1+cu118 requires Python 3.10-3.12" >&2
    exit 5
  fi
  if [[ -f "${VENV_DIR}/pyvenv.cfg" && -x "${VENV_DIR}/bin/python" ]]; then
    VENV_PYTHON_VERSION="$(${VENV_DIR}/bin/python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")' 2>/dev/null || true)"
    if [[ -n "${VENV_PYTHON_VERSION}" && "${VENV_PYTHON_VERSION}" != "${BASE_PYTHON_VERSION}" ]]; then
      echo "Existing venv uses Python ${VENV_PYTHON_VERSION}; recreating with Python ${BASE_PYTHON_VERSION}"
      rm -rf "${VENV_DIR}"
    fi
  fi
  "${BASE_PYTHON}" -m venv "${VENV_DIR}"
  # shellcheck disable=SC1091
  source "${VENV_DIR}/bin/activate"
  PYTHON_EXE="${VENV_DIR}/bin/python"

  echo
  echo "Python:"
  "${PYTHON_EXE}" -V | tee "${ENV_DIR}/python_version.txt"
  "${PYTHON_EXE}" -m pip install --upgrade pip setuptools wheel
  "${PYTHON_EXE}" -m pip install -r requirements-jobb.txt
  install_optional_hpsv2_if_requested

  "${PYTHON_EXE}" -m pip freeze > "${ENV_DIR}/pip_freeze.txt" || true
  "${PYTHON_EXE}" -m pip list > "${ENV_DIR}/pip_list.txt" || true

  "${PYTHON_EXE}" - <<'PY' > "${ENV_DIR}/package_versions.json"
import importlib.metadata as md
import json
import platform

packages = {
    "torch": "torch",
    "torchvision": "torchvision",
    "diffusers": "diffusers",
    "transformers": "transformers",
    "accelerate": "accelerate",
    "safetensors": "safetensors",
    "huggingface_hub": "huggingface-hub",
    "open_clip": "open-clip-torch",
    "lpips": "lpips",
    "hpsv2": "hpsv2",
    "numpy": "numpy",
    "matplotlib": "matplotlib",
    "tqdm": "tqdm",
    "PIL": "Pillow",
}

out = {"python": platform.python_version(), "packages": {}}
for import_name, dist_name in packages.items():
    try:
        out["packages"][dist_name] = md.version(dist_name)
    except Exception as exc:
        out["packages"][dist_name] = {"missing": True, "error": f"{type(exc).__name__}: {exc}"}

try:
    import torch
    out["torch_cuda"] = {
        "torch_cuda_version": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count(),
        "devices": [
            {
                "index": i,
                "name": torch.cuda.get_device_name(i),
                "total_gib": round(torch.cuda.get_device_properties(i).total_memory / 1024**3, 3),
            }
            for i in range(torch.cuda.device_count())
        ],
    }
except Exception as exc:
    out["torch_cuda"] = {"error": f"{type(exc).__name__}: {exc}"}

print(json.dumps(out, ensure_ascii=False, indent=2))
PY

  echo
  echo "Starting quantize-vector-jobb.py"
  "${PYTHON_EXE}" -u quantize-vector-jobb.py
}

install_optional_hpsv2_if_requested() {
  case "${RVQ_COMPUTE_HPS:-0}" in
    1|true|TRUE|yes|YES|y|Y|on|ON)
      ;;
    *)
      echo "RVQ_COMPUTE_HPS=${RVQ_COMPUTE_HPS:-0}; skipping optional hpsv2 installation"
      return 0
      ;;
  esac

  if "${PYTHON_EXE}" - <<'PY' >/dev/null 2>&1
import importlib.util
raise SystemExit(0 if importlib.util.find_spec("hpsv2") is not None else 1)
PY
  then
    echo "Optional hpsv2 package is already installed"
    return 0
  fi

  echo "RVQ_COMPUTE_HPS=1; installing optional hpsv2 package"
  "${PYTHON_EXE}" -m pip install "git+https://github.com/tgxs002/HPSv2.git"
}

select_idle_gpus() {
  # If the caller already set CUDA_VISIBLE_DEVICES, respect it and only align
  # RVQ_AUTO_MAX_DEVICES with the visible device count by default.
  if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    SELECTED_IDLE_GPU_COUNT="$(awk -F',' '{ count=0; for (i=1; i<=NF; i++) if ($i != "") count++; print count }' <<<"${CUDA_VISIBLE_DEVICES}")"
    export SELECTED_IDLE_GPU_COUNT
    echo "CUDA_VISIBLE_DEVICES is preset; using caller-selected GPUs: ${CUDA_VISIBLE_DEVICES}"
    return 0
  fi

  if ! command -v nvidia-smi >/dev/null 2>&1; then
    SELECTED_IDLE_GPU_COUNT=1
    export SELECTED_IDLE_GPU_COUNT
    echo "nvidia-smi not found; leaving CUDA_VISIBLE_DEVICES unset"
    return 0
  fi

  local max_used_mem_mib
  max_used_mem_mib="${RVQ_IDLE_GPU_MAX_USED_MEM_MIB:-2048}"

  local idle_gpus
  idle_gpus="$({
    nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader,nounits 2>/dev/null || true
  } | awk -F',' -v max_mem="${max_used_mem_mib}" '
      {
        gsub(/^[[:space:]]+|[[:space:]]+$/, "", $1)
        gsub(/^[[:space:]]+|[[:space:]]+$/, "", $2)
        gsub(/^[[:space:]]+|[[:space:]]+$/, "", $3)
        if ($2 == "0" && $3 <= max_mem) print $1
      }' | paste -sd, -)"

  if [[ -z "${idle_gpus}" ]]; then
    echo "ERROR: no GPUs with 0% utilization and <=${max_used_mem_mib} MiB used memory were found; refusing to start on busy GPUs" >&2
    exit 6
  fi

  export CUDA_VISIBLE_DEVICES="${idle_gpus}"
  SELECTED_IDLE_GPU_COUNT="$(awk -F',' '{ print NF }' <<<"${idle_gpus}")"
  export SELECTED_IDLE_GPU_COUNT
  echo "Selected idle GPUs with 0% utilization and <=${max_used_mem_mib} MiB used memory: ${CUDA_VISIBLE_DEVICES}"
}

finalize() {
  local status="$1"
  set +e
  echo
  echo "Finalizing JobB bundle with status=${status}"
  echo "${status}" > "${BUNDLE_DIR}/exit_code.txt"
  date -Is | tee "${BUNDLE_DIR}/finished_at.txt"

  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi > "${ENV_DIR}/nvidia-smi-final.txt"
  fi

  if [[ -f "${RESULTS_DIR}/rvq_jobb_results.json" ]]; then
    echo "Found JobB JSON: ${RESULTS_DIR}/rvq_jobb_results.json"
  else
    echo "WARNING: JobB JSON was not found in ${RESULTS_DIR}" >&2
  fi

  tar -czf "${ARCHIVE_PATH}" -C "$(dirname "${BUNDLE_DIR}")" "$(basename "${BUNDLE_DIR}")"
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "${ARCHIVE_PATH}" > "${ARCHIVE_PATH}.sha256"
  fi
  echo "Archive ready: ${ARCHIVE_PATH}"
}

FINALIZED=0
finalize_once() {
  local status="$1"
  if [[ "${FINALIZED}" == "1" ]]; then
    return 0
  fi
  FINALIZED=1
  finalize "${status}" 2>&1 | tee -a "${LOG_DIR}/run.log"
}

on_exit() {
  local status="$?"
  finalize_once "${status}"
}

trap on_exit EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

set +e
(
  set -Eeuo pipefail
  main
) 2>&1 | tee "${LOG_DIR}/run.log"
status="${PIPESTATUS[0]}"
set -e
finalize_once "${status}"
exit "${status}"
