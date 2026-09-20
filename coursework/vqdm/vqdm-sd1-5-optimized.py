# -*- coding: utf-8 -*-
# Auto-generated from vqdm-sd1-5.ipynb, then optimized for auto GPU runs.
# This is the canonical VQDM runner; hardware-specific wrappers should delegate here.

# %% [markdown] cell 0
# <h1>Импорты и конфиги</h1>

# %% cell 1
# === Kaggle/local setup ===
import os
import sys
import site
import shutil
import subprocess
import importlib.util
import time
import atexit
import builtins
import logging
import gc
from pathlib import Path
from urllib.request import urlopen

os.environ.setdefault("PYTHONNOUSERSITE", "1")


def _drop_user_site_packages_from_sys_path():
    """Keep conda/venv imports from being shadowed by ~/.local packages."""
    allow_user_site = os.environ.get("ALLOW_USER_SITE", "0").strip().lower() in {"1", "true", "yes", "y", "on"}
    if allow_user_site:
        return []

    candidates = []
    try:
        user_site = site.getusersitepackages()
        if isinstance(user_site, str):
            candidates.append(user_site)
        else:
            candidates.extend(user_site)
    except Exception:
        pass
    try:
        candidates.append(str(Path.home() / ".local"))
    except Exception:
        pass

    candidate_paths = []
    for item in candidates:
        try:
            candidate_paths.append(Path(item).expanduser().resolve())
        except Exception:
            pass

    kept = []
    removed = []
    for entry in sys.path:
        if not entry:
            kept.append(entry)
            continue
        try:
            resolved = Path(entry).expanduser().resolve()
        except Exception:
            kept.append(entry)
            continue
        if any(resolved == base or base in resolved.parents for base in candidate_paths):
            removed.append(entry)
        else:
            kept.append(entry)
    sys.path[:] = kept
    return removed


_REMOVED_USER_SITE_PATHS = _drop_user_site_packages_from_sys_path()

SCRIPT_STARTED_AT = time.time()
_ORIGINAL_PRINT = builtins.print


def _print_with_flush(*args, **kwargs):
    kwargs.setdefault("flush", True)
    return _ORIGINAL_PRINT(*args, **kwargs)


builtins.print = _print_with_flush

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
LOGGER = logging.getLogger("vqdm-sd1-5")


def _format_seconds(seconds: float) -> str:
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h:d}h {m:02d}m {s:02d}s"
    if m:
        return f"{m:d}m {s:02d}s"
    return f"{s:d}s"


def log_event(message: str):
    LOGGER.info(message)


def log_section(title: str):
    line = "=" * 110
    print()
    print(line)
    LOGGER.info(title)
    print(line)


def _finish_logging():
    LOGGER.info("Run finished in %s", _format_seconds(time.time() - SCRIPT_STARTED_AT))


def _log_uncaught_exception(exc_type, exc, tb):
    LOGGER.critical("Uncaught exception", exc_info=(exc_type, exc, tb))


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    return int(value)


def _env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    return float(value)


atexit.register(_finish_logging)
sys.excepthook = _log_uncaught_exception
log_section("Script start")

IN_KAGGLE = "KAGGLE_KERNEL_RUN_TYPE" in os.environ
print("Kaggle detected:" if IN_KAGGLE else "Local/other runtime detected:", IN_KAGGLE)
if _REMOVED_USER_SITE_PATHS:
    print("Ignored user-site package paths:", _REMOVED_USER_SITE_PATHS)
VQDM_COMPUTE_HPS_REQUESTED = _env_bool("VQDM_COMPUTE_HPS", False)

_REQUIRED_PACKAGES = [
    ("numpy", "numpy"),
    ("matplotlib", "matplotlib"),
    ("tqdm", "tqdm"),
    ("PIL", "Pillow"),
    ("diffusers", "diffusers"),
    ("transformers", "transformers"),
    ("accelerate", "accelerate"),
    ("safetensors", "safetensors"),
    ("open_clip", "open_clip_torch"),
    ("lpips", "lpips"),
    ("huggingface_hub", "huggingface_hub"),
]
if VQDM_COMPUTE_HPS_REQUESTED:
    _REQUIRED_PACKAGES.append(("hpsv2", "hpsv2"))

missing = [pkg for module_name, pkg in _REQUIRED_PACKAGES if importlib.util.find_spec(module_name) is None]
if missing:
    print("Installing missing packages:", missing)
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *missing])
    except Exception as exc:
        print("Package installation failed. In Kaggle, enable Internet or attach a dataset/wheel cache with these packages.")
        raise
else:
    print("All required packages are already importable.")


def _get_kaggle_or_env_secret(name: str):
    value = os.environ.get(name)
    if value:
        return value
    try:
        from kaggle_secrets import UserSecretsClient
        return UserSecretsClient().get_secret(name)
    except Exception:
        return None


hf_token = _get_kaggle_or_env_secret("HF_TOKEN")
if hf_token:
    os.environ["HF_TOKEN"] = hf_token
    os.environ["HUGGINGFACE_HUB_TOKEN"] = hf_token
    try:
        from huggingface_hub import login
        login(token=hf_token)
        print("Hugging Face token loaded from Kaggle Secrets/env.")
    except Exception as exc:
        print("HF login skipped:", type(exc).__name__, exc)
else:
    print("HF_TOKEN is not set; using cached Hugging Face auth or public model access.")


def _download_file_with_retries(
    url: str,
    dst: Path,
    attempts: int = 3,
    sleep_s: float = 3.0,
    timeout_s: float = 20.0,
) -> bool:
    last_exc = None
    for attempt in range(1, attempts + 1):
        try:
            with urlopen(url, timeout=timeout_s) as response, open(dst, "wb") as f:
                shutil.copyfileobj(response, f)
            return True
        except Exception as exc:
            last_exc = exc
            print(f"Download failed ({attempt}/{attempts}): {type(exc).__name__}: {exc}")
            try:
                if dst.exists():
                    dst.unlink()
            except Exception:
                pass
            if attempt < attempts:
                time.sleep(sleep_s)
    print(f"Download failed permanently: {type(last_exc).__name__}: {last_exc}")
    return False


def ensure_hpsv2_bpe():
    filename = "bpe_simple_vocab_16e6.txt.gz"
    spec = importlib.util.find_spec("hpsv2")
    if spec is None or not spec.submodule_search_locations:
        print("hpsv2 is not importable; skip BPE placement.")
        return False

    hps_root = Path(list(spec.submodule_search_locations)[0])
    candidate_dirs = [
        hps_root / "src" / "open_clip",
        hps_root / "open_clip",
    ]
    for base in list(site.getsitepackages()) + [site.getusersitepackages()]:
        candidate_dirs.append(Path(base) / "hpsv2" / "src" / "open_clip")

    for d in candidate_dirs:
        if (d / filename).exists():
            print("HPSv2 BPE already exists:", d / filename)
            return True

    work_dir = Path("/kaggle/working") if IN_KAGGLE else Path.cwd()
    local_bpe = work_dir / filename

    if not local_bpe.exists() and IN_KAGGLE:
        input_root = Path("/kaggle/input")
        if input_root.exists():
            matches = list(input_root.rglob(filename))
            if matches:
                print("Using uploaded CLIP BPE:", matches[0])
                shutil.copyfile(matches[0], local_bpe)

    if not local_bpe.exists() and IN_KAGGLE:
        print("Skipping online HPSv2 BPE download on Kaggle; HPS metrics will be skipped/NaN.")
        return False

    if not local_bpe.exists():
        print("Downloading CLIP BPE for hpsv2...")
        ok = _download_file_with_retries(
            "https://openaipublic.azureedge.net/clip/bpe_simple_vocab_16e6.txt.gz",
            local_bpe,
            attempts=3,
            sleep_s=3.0,
            timeout_s=20.0,
        )
        if not ok:
            print("HPSv2 BPE is unavailable; HPS metrics will be skipped/NaN, but the notebook can continue.")
            return False

    placed = []
    for d in candidate_dirs:
        try:
            d.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(local_bpe, d / filename)
            placed.append(str(d / filename))
        except Exception:
            pass

    if placed:
        print("Placed HPSv2 BPE into:", placed[0])
        return True

    print("Could not place HPSv2 BPE into hpsv2/open_clip; HPS metrics will be skipped/NaN.")
    return False


HPSV2_BPE_READY = ensure_hpsv2_bpe() if VQDM_COMPUTE_HPS_REQUESTED else False
log_event(f"HPSv2 BPE ready: {HPSV2_BPE_READY}")


def _select_idle_cuda_visible_devices_for_vqdm():
    if not _env_bool("VQDM_SELECT_IDLE_GPUS", True):
        print("VQDM_SELECT_IDLE_GPUS=0; leaving CUDA_VISIBLE_DEVICES unchanged.")
        return
    if os.environ.get("CUDA_VISIBLE_DEVICES"):
        print(f"CUDA_VISIBLE_DEVICES is preset; using caller-selected GPUs: {os.environ['CUDA_VISIBLE_DEVICES']}")
        return
    if os.environ.get("VQDM_DEVICES") or os.environ.get("VQDM_MULTI_GPU_DEVICES"):
        print("VQDM_DEVICES/VQDM_MULTI_GPU_DEVICES is preset; skipping automatic idle-GPU selection.")
        return

    nvidia_smi = shutil.which("nvidia-smi")
    if not nvidia_smi:
        print("nvidia-smi not found; leaving CUDA_VISIBLE_DEVICES unset.")
        return

    max_used_mem_mib = _env_int("VQDM_IDLE_GPU_MAX_USED_MEM_MIB", 2048)
    try:
        out = subprocess.check_output(
            [
                nvidia_smi,
                "--query-gpu=index,utilization.gpu,memory.used",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except Exception as exc:
        print(f"Could not query nvidia-smi for idle GPUs: {type(exc).__name__}: {exc}")
        return

    idle_ids = []
    for line in out.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 3:
            continue
        try:
            gpu_idx = int(parts[0])
            util_pct = int(parts[1])
            used_mem_mib = int(parts[2])
        except ValueError:
            continue
        if util_pct == 0 and used_mem_mib <= max_used_mem_mib:
            idle_ids.append(str(gpu_idx))

    if not idle_ids:
        raise RuntimeError(
            f"No GPUs with 0% utilization and <={max_used_mem_mib} MiB used memory were found; "
            "set CUDA_VISIBLE_DEVICES manually or VQDM_SELECT_IDLE_GPUS=0 to override."
        )

    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(idle_ids)
    print(
        f"Selected idle GPUs with 0% utilization and <={max_used_mem_mib} MiB used memory: "
        f"{os.environ['CUDA_VISIBLE_DEVICES']}"
    )


_select_idle_cuda_visible_devices_for_vqdm()

# %% cell 2
import os
import math
import copy
import time
import random
import contextlib
import functools
import json
import warnings
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Optional, List, Dict, Sequence, Tuple, Union, Any
from pathlib import Path
import csv
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel.scatter_gather import Gather
from tqdm.auto import tqdm, trange

from diffusers import StableDiffusionPipeline, DDIMScheduler


def _is_cuda_oom(exc: BaseException) -> bool:
    msg = str(exc).lower()
    return isinstance(exc, torch.cuda.OutOfMemoryError) or (
        isinstance(exc, RuntimeError)
        and ("out of memory" in msg or "cuda error: out of memory" in msg)
    )


def _clear_cuda_after_oom():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass


def _cuda_mem_info(device=None):
    if not torch.cuda.is_available():
        return None
    if device is None:
        device = torch.cuda.current_device()
    with torch.cuda.device(device):
        free, total = torch.cuda.mem_get_info()
    return {
        "free": int(free),
        "total": int(total),
        "free_gib": free / 1024**3,
        "total_gib": total / 1024**3,
    }


def _cuda_total_gib(device=None, default: float = 0.0) -> float:
    info = _cuda_mem_info(device)
    return default if info is None else float(info["total_gib"])


def _is_big_gpu(min_total_gib: float = 40.0) -> bool:
    return torch.cuda.is_available() and _cuda_total_gib() >= min_total_gib


def log_cuda_memory(label: str = "", device=None):
    try:
        if not torch.cuda.is_available():
            return
        info = _cuda_mem_info(device)
        idx = torch.cuda.current_device() if device is None else torch.device(device).index
        allocated = torch.cuda.memory_allocated(idx)
        reserved = torch.cuda.memory_reserved(idx)
        suffix = f" | {label}" if label else ""
        LOGGER.info(
            "CUDA memory%s | cuda:%s allocated=%.2f GiB reserved=%.2f GiB free=%.2f GiB total=%.2f GiB",
            suffix,
            idx,
            allocated / 1024**3,
            reserved / 1024**3,
            info["free_gib"],
            info["total_gib"],
        )
    except Exception as exc:
        LOGGER.info("CUDA memory logging skipped: %s: %s", type(exc).__name__, exc)


def _parse_cuda_device_ids(value: Optional[str]) -> Optional[List[int]]:
    if value is None or value.strip() == "":
        return None
    ids = []
    for part in value.replace(";", ",").split(","):
        part = part.strip()
        if not part:
            continue
        if part.startswith("cuda:"):
            part = part.split(":", 1)[1]
        ids.append(int(part))
    return ids or None


def _cuda_device_totals_gib() -> List[float]:
    if not torch.cuda.is_available():
        return []
    totals = []
    for idx in range(torch.cuda.device_count()):
        try:
            totals.append(_cuda_total_gib(idx))
        except Exception:
            totals.append(0.0)
    return totals


def _device_vram_weight(device: torch.device) -> float:
    info = _cuda_mem_info(device)
    if info is None:
        return 1.0
    # Free VRAM drives the split, with a total-VRAM floor so smaller idle cards
    # still get useful work.
    return max(float(info.get("free_gib", 0.0)), float(info.get("total_gib", 0.0)) * 0.25, 1.0)


def _weighted_slices(n: int, weights: Sequence[float]) -> List[slice]:
    n = max(int(n), 0)
    weights = [max(float(w), 1e-6) for w in weights]
    if n <= 0 or not weights:
        return []
    if len(weights) == 1:
        return [slice(0, n)]

    if n >= len(weights):
        total = sum(weights)
        counts = [1 for _ in weights]
        remaining = n - len(weights)
        raw = [remaining * w / total for w in weights]
        counts = [count + int(math.floor(extra)) for count, extra in zip(counts, raw)]
        remainder = n - sum(counts)
        order = sorted(range(len(weights)), key=lambda i: raw[i] - math.floor(raw[i]), reverse=True)
        for idx in order[:remainder]:
            counts[idx] += 1
    else:
        # Callers trim their device lists to the number of returned slices, so
        # tiny jobs must stay aligned with the leading devices.
        counts = [1 if idx < n else 0 for idx in range(len(weights))]

    slices = []
    start = 0
    for count in counts:
        stop = min(n, start + count)
        if start < stop:
            slices.append(slice(start, stop))
        start = stop
    return slices

# %% cell 3
# === Runtime/device setup ===
_AUTO_CUDA_TOTALS_GIB = _cuda_device_totals_gib()
_AUTO_MAX_CUDA_GIB = max(_AUTO_CUDA_TOTALS_GIB, default=0.0)
_AUTO_DEVICE_LIST_ENV = ",".join(str(i) for i in range(len(_AUTO_CUDA_TOTALS_GIB))) or None
VQDM_HW_PROFILE = os.environ.get("VQDM_HW_PROFILE", "auto").strip().lower()
if VQDM_HW_PROFILE not in {"auto", "h100", "v100x4", "balanced", "cpu"}:
    raise ValueError("VQDM_HW_PROFILE must be one of auto, h100, v100x4, balanced, cpu")
if VQDM_HW_PROFILE == "auto":
    if not torch.cuda.is_available():
        VQDM_HW_PROFILE_RESOLVED = "cpu"
    elif _AUTO_MAX_CUDA_GIB < 48.0:
        VQDM_HW_PROFILE_RESOLVED = "v100x4"
    else:
        VQDM_HW_PROFILE_RESOLVED = "h100"
else:
    VQDM_HW_PROFILE_RESOLVED = VQDM_HW_PROFILE

if VQDM_HW_PROFILE_RESOLVED == "v100x4":
    _VQDM_DEFAULT_XTX_MAX_BATCH = 8
    _VQDM_DEFAULT_CONV_XTX_MAX_BATCH = 2
    _VQDM_DEFAULT_GEN_BATCH_MAX = 8
    _VQDM_DEFAULT_TARGET_MEM_FRACTION = 0.58
    _VQDM_DEFAULT_HIGH_VRAM_THRESHOLD_GIB = 24.0
    _VQDM_DEFAULT_XTX_GIB_PER_PROMPT = 1.75
elif VQDM_HW_PROFILE_RESOLVED == "balanced":
    _VQDM_DEFAULT_XTX_MAX_BATCH = 8 if _AUTO_MAX_CUDA_GIB < 48.0 else 16
    _VQDM_DEFAULT_CONV_XTX_MAX_BATCH = 2 if _AUTO_MAX_CUDA_GIB < 48.0 else 4
    _VQDM_DEFAULT_GEN_BATCH_MAX = 8 if _AUTO_MAX_CUDA_GIB < 48.0 else 16
    _VQDM_DEFAULT_TARGET_MEM_FRACTION = 0.62
    _VQDM_DEFAULT_HIGH_VRAM_THRESHOLD_GIB = 24.0 if _AUTO_MAX_CUDA_GIB < 48.0 else 40.0
    _VQDM_DEFAULT_XTX_GIB_PER_PROMPT = 1.50
else:
    _VQDM_DEFAULT_XTX_MAX_BATCH = 0
    _VQDM_DEFAULT_CONV_XTX_MAX_BATCH = 4
    _VQDM_DEFAULT_GEN_BATCH_MAX = 0
    _VQDM_DEFAULT_TARGET_MEM_FRACTION = 0.70
    _VQDM_DEFAULT_HIGH_VRAM_THRESHOLD_GIB = 40.0
    _VQDM_DEFAULT_XTX_GIB_PER_PROMPT = 1.25

VQDM_GPU_AUTOTUNE = _env_bool("VQDM_GPU_AUTOTUNE", True)
VQDM_ALLOW_TF32 = _env_bool("VQDM_ALLOW_TF32", True)
VQDM_FAST_SMOKE_TEST = _env_bool("VQDM_FAST_SMOKE_TEST", False)
VQDM_SCOPE_SWEEP = _env_bool("VQDM_SCOPE_SWEEP", True)
VQDM_INCLUDE_CONV_ENV = _env_bool("VQDM_INCLUDE_CONV", False)
VQDM_TARGET_MODE = os.environ.get("VQDM_TARGET_MODE", "hardcoded_midblock").strip().lower()
VQDM_SINGLE_TARGET_LAYER = os.environ.get("VQDM_SINGLE_TARGET_LAYER", "").strip()
VQDM_RUN_ID = os.environ.get("VQDM_RUN_ID", time.strftime("%Y%m%d_%H%M%S"))
VQDM_OUTPUT_DIR_ENV = os.environ.get("VQDM_OUTPUT_DIR") or os.environ.get("VQDM_RESULTS_DIR")
VQDM_DEVICES_ENV = os.environ.get("VQDM_DEVICES") or os.environ.get("VQDM_MULTI_GPU_DEVICES") or _AUTO_DEVICE_LIST_ENV
VQDM_MULTI_GPU_XTX = _env_bool("VQDM_MULTI_GPU_XTX", True)
VQDM_MULTI_GPU_AQ = _env_bool("VQDM_MULTI_GPU_AQ", True)
VQDM_XTX_BATCH_ENV = _env_int("VQDM_XTX_BATCH", 0)  # 0 = auto
VQDM_XTX_BINSEARCH = _env_bool("VQDM_XTX_BINSEARCH", True)
VQDM_XTX_PROBE_STEPS = _env_int("VQDM_XTX_PROBE_STEPS", 2)
VQDM_XTX_MAX_BATCH = _env_int("VQDM_XTX_MAX_BATCH", _VQDM_DEFAULT_XTX_MAX_BATCH)  # 0 = uncapped by env
VQDM_CONV_XTX_MAX_BATCH = _env_int("VQDM_CONV_XTX_MAX_BATCH", _VQDM_DEFAULT_CONV_XTX_MAX_BATCH)
VQDM_GEN_BATCH_ENV = _env_int("VQDM_GEN_BATCH", 0)  # 0 = auto
VQDM_GEN_BATCH_MAX = _env_int("VQDM_GEN_BATCH_MAX", _VQDM_DEFAULT_GEN_BATCH_MAX)
VQDM_GPU_TARGET_MEM_FRACTION = _env_float("VQDM_GPU_TARGET_MEM_FRACTION", _VQDM_DEFAULT_TARGET_MEM_FRACTION)
VQDM_HIGH_VRAM_THRESHOLD_GIB = _env_float("VQDM_HIGH_VRAM_THRESHOLD_GIB", _VQDM_DEFAULT_HIGH_VRAM_THRESHOLD_GIB)
VQDM_XTX_ESTIMATED_GIB_PER_PROMPT = _env_float("VQDM_XTX_ESTIMATED_GIB_PER_PROMPT", _VQDM_DEFAULT_XTX_GIB_PER_PROMPT)
VQDM_CPU_COPY_BEFORE_PIPE_CLONE = _env_bool("VQDM_CPU_COPY_BEFORE_PIPE_CLONE", False)
VQDM_STORE_XTX_IN_QWEIGHT = _env_bool("VQDM_STORE_XTX_IN_QWEIGHT", False)
VQDM_METRIC_DEVICE_ENV = os.environ.get("VQDM_METRIC_DEVICE")

torch.backends.cudnn.allow_tf32 = bool(VQDM_ALLOW_TF32)
torch.backends.cuda.matmul.allow_tf32 = bool(VQDM_ALLOW_TF32)

runtime_device = "cuda" if torch.cuda.is_available() else "cpu"
runtime_dtype = torch.float16 if runtime_device == "cuda" else torch.float32

if torch.cuda.is_available():
    torch.cuda.empty_cache()
    try:
        torch.cuda.ipc_collect()
    except Exception:
        pass


def resolve_vqdm_devices() -> List[torch.device]:
    if runtime_device != "cuda":
        return [torch.device("cpu")]
    requested = _parse_cuda_device_ids(VQDM_DEVICES_ENV)
    available = list(range(torch.cuda.device_count()))
    ids = requested if requested is not None else available
    ids = [idx for idx in ids if idx in available]
    if not ids:
        ids = [torch.cuda.current_device()]
    return [torch.device(f"cuda:{idx}") for idx in ids]


vqdm_devices = resolve_vqdm_devices()
vqdm_primary_device = vqdm_devices[0]
vqdm_high_vram_gpu = runtime_device == "cuda" and _cuda_total_gib(vqdm_primary_device) >= VQDM_HIGH_VRAM_THRESHOLD_GIB
vqdm_fast_smoke_test = VQDM_FAST_SMOKE_TEST
vqdm_cpu_deepcopy_before_quant = _env_bool("VQDM_CPU_DEEPCOPY_BEFORE_QUANT", False if vqdm_high_vram_gpu else True)
vqdm_cpu_copy_before_pipe_clone = _env_bool("VQDM_CPU_COPY_BEFORE_PIPE_CLONE", False if vqdm_high_vram_gpu else True)
vqdm_eval_inplace_midblock = _env_bool("VQDM_EVAL_INPLACE_MIDBLOCK", True if vqdm_high_vram_gpu else False)

vqdm_collect_device = str(vqdm_primary_device)
vqdm_collect_devices = vqdm_devices if (VQDM_MULTI_GPU_XTX and len(vqdm_devices) > 1) else [vqdm_primary_device]
vqdm_xtx_store_device = os.environ.get("VQDM_XTX_STORE_DEVICE", str(vqdm_primary_device))
vqdm_quant_device = str(vqdm_primary_device)
vqdm_quant_devices = vqdm_devices if (VQDM_MULTI_GPU_AQ and len(vqdm_devices) > 1) else [vqdm_primary_device]
vqdm_final_device = str(vqdm_primary_device)
vqdm_final_dtype = runtime_dtype
vqdm_eval_device = str(vqdm_primary_device)
if VQDM_METRIC_DEVICE_ENV:
    vqdm_metric_device = VQDM_METRIC_DEVICE_ENV
elif len(vqdm_devices) > 1 and vqdm_high_vram_gpu:
    vqdm_metric_device = str(vqdm_devices[min(len(vqdm_devices) - 1, 3)])
else:
    vqdm_metric_device = str(vqdm_primary_device) if runtime_device == "cuda" else "cpu"
vqdm_keep_eval_pipes_on_device = _env_bool(
    "VQDM_KEEP_EVAL_PIPES_ON_DEVICE",
    True if vqdm_high_vram_gpu else (False if globals().get("IN_KAGGLE", False) else True),
)
vqdm_include_conv = VQDM_INCLUDE_CONV_ENV
vqdm_scope_sweep = bool(VQDM_SCOPE_SWEEP)
vqdm_results_dir = Path(VQDM_OUTPUT_DIR_ENV or f"runs/vqdm_pickscore_sweep_{VQDM_RUN_ID}")
vqdm_results_dir.mkdir(parents=True, exist_ok=True)
vqdm_results_json_path = vqdm_results_dir / "vqdm_pickscore_sweep_results.json"
vqdm_results_csv_path = vqdm_results_dir / "vqdm_pickscore_sweep_summary.csv"

device = str(vqdm_primary_device) if runtime_device == "cuda" else "cpu"
base_dtype = runtime_dtype

if runtime_device == "cuda":
    for dev in vqdm_devices:
        LOGGER.info(
            "CUDA device %s: %s, capability=sm_%s%s, total=%.1f GiB",
            dev,
            torch.cuda.get_device_name(dev),
            *torch.cuda.get_device_capability(dev),
            _cuda_total_gib(dev),
        )

log_section("Load Stable Diffusion")
log_event(
    f"runtime_device={runtime_device}, runtime_dtype={runtime_dtype}, devices={[str(d) for d in vqdm_devices]}, "
    f"metric_device={vqdm_metric_device}, target_mode={VQDM_TARGET_MODE}, include_conv={vqdm_include_conv}, "
    f"scope_sweep={vqdm_scope_sweep}, output_dir={vqdm_results_dir}, "
    f"hw_profile={VQDM_HW_PROFILE_RESOLVED} (requested={VQDM_HW_PROFILE}), "
    f"cuda_totals_gib={[round(x, 1) for x in _AUTO_CUDA_TOTALS_GIB]}, "
    f"gpu_autotune={VQDM_GPU_AUTOTUNE}, xtx_binsearch={VQDM_XTX_BINSEARCH}, allow_tf32={VQDM_ALLOW_TF32}, "
    f"eval_inplace_midblock={vqdm_eval_inplace_midblock}, keep_eval_pipes_on_device={vqdm_keep_eval_pipes_on_device}, "
    f"cpu_deepcopy_before_quant={vqdm_cpu_deepcopy_before_quant}, cpu_copy_before_pipe_clone={vqdm_cpu_copy_before_pipe_clone}, "
    f"target_mem_fraction={VQDM_GPU_TARGET_MEM_FRACTION}, xtx_max_batch={VQDM_XTX_MAX_BATCH}, "
    f"conv_xtx_max_batch={VQDM_CONV_XTX_MAX_BATCH}, gen_batch_max={VQDM_GEN_BATCH_MAX}"
)
log_cuda_memory("before model load", vqdm_primary_device)

pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=base_dtype,
    safety_checker=None,
).to(device)

# Official VQDM text2image quantization uses DDIM with guidance_scale=5.0.
pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
pipe.set_progress_bar_config(disable=False)

print("device:", device)
print("dtype:", base_dtype)
print("scheduler:", type(pipe.scheduler).__name__)
print("unet dtype:", next(pipe.unet.parameters()).dtype)
print(
    f"collect={vqdm_collect_device}, xtx={vqdm_xtx_store_device}, quant={vqdm_quant_device}, "
    f"final={vqdm_final_device}, metrics={vqdm_metric_device}, quant_devices={[str(d) for d in vqdm_quant_devices]}"
)
log_cuda_memory("after Stable Diffusion load", vqdm_primary_device)


def force_pipeline_device_dtype(pipe, device, dtype):
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pipe = pipe.to(device, torch_dtype=dtype)

    for component_name in ("unet", "vae", "text_encoder", "safety_checker"):
        component = getattr(pipe, component_name, None)
        if isinstance(component, nn.Module):
            component.to(device=device, dtype=dtype)
    return pipe


def move_pipeline_to_cpu_safely(pipe):
    return force_pipeline_device_dtype(pipe, "cpu", torch.float32)


def optimize_pipeline_for_inference(pipe_obj, label: str = "pipeline"):
    """Apply the same low-risk inference optimizations used by the RVQ runner."""
    try:
        pipe_obj.set_progress_bar_config(disable=True)
    except Exception:
        pass
    try:
        pipe_obj.unet.eval()
        pipe_obj.vae.eval()
        pipe_obj.text_encoder.eval()
    except Exception:
        pass
    if torch.cuda.is_available():
        try:
            pipe_obj.unet.to(memory_format=torch.channels_last)
            pipe_obj.vae.to(memory_format=torch.channels_last)
            log_event(f"Inference optimization applied to {label}: channels_last for UNet/VAE")
        except Exception as exc:
            log_event(f"Inference optimization skipped for {label}: {type(exc).__name__}: {exc}")
    try:
        if hasattr(pipe_obj, "disable_attention_slicing"):
            pipe_obj.disable_attention_slicing()
    except Exception:
        pass
    return pipe_obj


optimize_pipeline_for_inference(pipe, label="FP pipeline initial")
log_cuda_memory("after FP pipeline inference optimization", vqdm_primary_device)

# %% cell 4
@dataclass
class AQConfig:
    # grouping
    in_group_size: int = 8
    out_group_size: int = 1

    # additive quantization
    num_codebooks: int = 4
    nbits_per_codebook: int = 8
    code_dtype: torch.dtype = torch.int32

    # optional extra compression
    groupwise_codebooks: bool = False
    input_channel_scales: bool = False
    codebook_value_nbits: int = 16
    codebook_value_num_groups: int = 1
    scale_nbits: int = 0

    # init
    init_max_iter: int = 100
    init_max_points_per_centroid: Optional[int] = None
    use_faiss: bool = False

    # optimization
    lr: float = 1e-4
    beam_size: int = 8
    max_epochs: int = 1000
    steps_per_epoch: int = 100
    relative_mse_tolerance: Optional[float] = 1e-2

    # misc
    seed: int = 0
    accum_dtype: torch.dtype = torch.float64
    quant_device: str = "auto"
    devices: Optional[Sequence[torch.device]] = None

# %% [markdown] cell 5
# <h1>Вспомогательные функции</h1>

# %% cell 6
ellipsis = type(...)

@functools.lru_cache()
def maybe_script(fn):
    using_tpu = bool(os.environ.get("TPU_NAME"))
    should_script = int(os.environ.get("AQ_USE_JIT", not using_tpu))
    return torch.jit.script(fn) if should_script else fn


def is_signed(dtype: torch.dtype) -> bool:
    try:
        return dtype.is_signed
    except RuntimeError:
        if dtype.is_floating_point:
            return torch.finfo(dtype).min < 0
        else:
            return torch.iinfo(dtype).min < 0 and torch.iinfo(dtype).max > 0

# %% cell 7
def find_quantizable_sublayers(module: nn.Module) -> Dict[str, nn.Module]:
    res = {}
    for name, layer in module.named_modules():
        if isinstance(layer, (nn.Linear, nn.Conv2d)):
            res[name] = layer
    return res


def is_vqdm_default_mid_block_layer(
    name: str,
    layer: nn.Module,
    min_features: int = 16,
    include_linear: bool = True,
    include_conv: bool = True,
) -> bool:
    """
    VQDM-style filter for one SD mid_block.
    Matches the paper/code default: attention/FF linear layers and ResNet conv layers,
    while skipping time embedding projections.
    """
    if "time_emb_proj" in name:
        return False

    if include_linear and isinstance(layer, nn.Linear):
        if min(layer.in_features, layer.out_features) < min_features:
            return False
        return name.endswith((
            "to_q",
            "to_k",
            "to_v",
            "to_out.0",
            "ff.net.0.proj",
            "ff.net.2",
            "proj_in",
            "proj_out",
        ))

    if include_conv and isinstance(layer, nn.Conv2d):
        if min(layer.in_channels, layer.out_channels) < min_features:
            return False
        return name.endswith(("conv1", "conv2", "conv_shortcut", "proj_in", "proj_out"))

    return False


def get_mid_block_quant_layers(
    unet: nn.Module,
    min_features: int = 16,
    include_linear: bool = True,
    include_conv: bool = True,
) -> Dict[str, nn.Module]:
    res = {}

    for name, layer in unet.mid_block.named_modules():
        if is_vqdm_default_mid_block_layer(
            name,
            layer,
            min_features=min_features,
            include_linear=include_linear,
            include_conv=include_conv,
        ):
            full_name = f"mid_block.{name}" if name else "mid_block"
            res[full_name] = layer

    return res


def get_all_mid_block_quant_layers(
    unet: nn.Module,
    min_features: int = 16,
    include_linear: bool = True,
    include_conv: bool = True,
) -> Dict[str, nn.Module]:
    """Return every Linear/Conv2d in UNet.mid_block, including time_emb_proj."""
    res = {}
    for name, layer in unet.mid_block.named_modules():
        if not name:
            continue
        full_name = f"mid_block.{name}"
        if include_linear and isinstance(layer, nn.Linear):
            if min(layer.in_features, layer.out_features) >= min_features:
                res[full_name] = layer
        elif include_conv and isinstance(layer, nn.Conv2d):
            if min(layer.in_channels, layer.out_channels) >= min_features:
                res[full_name] = layer
    return res

# %% cell 8
def get_module_by_name(root: nn.Module, name: str) -> nn.Module:
    cur = root
    for part in name.split("."):
        cur = getattr(cur, part)
    return cur


def set_module_by_name(root: nn.Module, name: str, new_module: nn.Module):
    parts = name.split(".")
    parent = root
    for part in parts[:-1]:
        parent = getattr(parent, part)
    setattr(parent, parts[-1], new_module)

# %% cell 9
def print_mid_block_quant_layers(
    unet: nn.Module,
    min_features: int = 16,
    include_linear: bool = True,
    include_conv: bool = True,
    all_linear_conv: bool = False,
):
    layer_getter = get_all_mid_block_quant_layers if all_linear_conv else get_mid_block_quant_layers
    layers = layer_getter(unet, min_features=min_features, include_linear=include_linear, include_conv=include_conv)
    print(f"Found {len(layers)} quantizable layers in mid_block")
    for name, layer in layers.items():
        if isinstance(layer, nn.Linear):
            print(f"{name:60s}  Linear  weight={tuple(layer.weight.shape)}")
        else:
            print(f"{name:60s}  Conv2d  weight={tuple(layer.weight.shape)}")
    return layers

# %% [markdown] cell 10
# <h1>K-means</h1>

# %% cell 11
@maybe_script
def _kmeans_greedy_init(data: torch.Tensor, k: int) -> torch.Tensor:
    clusters = torch.zeros(k, data.shape[1], device=data.device, dtype=data.dtype)
    running_min_distances = torch.full((data.shape[0],), torch.inf, device=data.device, dtype=data.dtype)
    data_norm_squared = data.norm(p=2, dim=1).square()

    for i in range(k):
        clusters[i] = data[running_min_distances.argmax()]
        distances_to_cluster_i = data_norm_squared - 2 * data @ clusters[i] + clusters[i].norm().square()
        running_min_distances = torch.minimum(running_min_distances, distances_to_cluster_i, out=running_min_distances)

    return clusters

# %% cell 12
@maybe_script
def fit_kmeans(
    data: torch.Tensor,
    k: int,
    max_iter: int = 1000,
    check_every: int = 10,
    rtol: float = 1e-6,
    atol: float = 1e-8,
    greedy_init: bool = False,
    block_size_vals: int = 2**30,
):
    """
    data: [N, D]
    returns:
        clusters: [k, D]
        nearest_indices: [N]
        reconstructed_data: [N, D]
    """
    if greedy_init:
        clusters = _kmeans_greedy_init(data, k)
    else:
        perm = torch.randperm(data.shape[0], device=data.device)
        clusters = data[perm[:k], :].clone()

    block_size = max(1, block_size_vals // k)
    nearest_indices = torch.empty(len(data), dtype=torch.int64, device=data.device)

    for i in range(max_iter):
        for block_start in range(0, len(data), block_size):
            block = data[block_start : block_start + block_size]
            nearest_indices[block_start : block_start + block_size] = torch.addmm(
                torch.bmm(clusters[:, None, :], clusters[:, :, None]).flatten(),
                block,
                clusters.T,
                beta=-0.5,
            ).argmax(1)

        new_clusters = (
            clusters.clone()
            .index_reduce_(dim=0, index=nearest_indices, source=data, reduce="mean", include_self=False)
        )

        if i % check_every == 0:
            if torch.allclose(new_clusters, clusters, rtol=rtol, atol=atol):
                clusters = new_clusters
                break

        clusters = new_clusters

    for block_start in range(0, len(data), block_size):
        block = data[block_start : block_start + block_size]
        nearest_indices[block_start : block_start + block_size] = torch.addmm(
            torch.bmm(clusters[:, None, :], clusters[:, :, None]).flatten(),
            block,
            clusters.T,
            beta=-0.5,
        ).argmax(1)

    reconstructed_data = clusters[nearest_indices]
    return clusters, nearest_indices, reconstructed_data

# %% cell 13
def fit_faiss_kmeans(
    data: torch.Tensor,
    k: int,
    *,
    max_iter: int = 1000,
    gpu: bool = True,
    max_points_per_centroid: Optional[int] = None,
    verbose: bool = True,
):
    import faiss

    d = data.shape[1]
    if max_points_per_centroid is not None:
        kmeans = faiss.Kmeans(
            d, k, niter=max_iter, verbose=verbose, gpu=gpu, max_points_per_centroid=max_points_per_centroid
        )
    else:
        kmeans = faiss.Kmeans(d, k, niter=max_iter, verbose=verbose, gpu=gpu)

    kmeans.train(data.detach().cpu().numpy())
    clusters = torch.from_numpy(kmeans.centroids).to(data.device, dtype=data.dtype)
    nearest_indices = torch.from_numpy(kmeans.index.search(data.detach().cpu().numpy(), 1)[1][:, 0]).to(data.device)
    reconstructed_data = clusters[nearest_indices]
    return clusters, nearest_indices, reconstructed_data

# %% cell 14
@maybe_script
def find_nearest_cluster(data: torch.Tensor, clusters: torch.Tensor, block_size_vals: int = 2**30):
    block_size = max(1, block_size_vals // len(clusters))
    nearest_indices = torch.empty(len(data), dtype=torch.int64, device=data.device)

    for block_start in range(0, len(data), block_size):
        block = data[block_start : block_start + block_size]
        nearest_indices[block_start : block_start + block_size] = torch.addmm(
            torch.bmm(clusters[:, None, :], clusters[:, :, None]).flatten(),
            block,
            clusters.T,
            beta=-0.5,
        ).argmax(1)

    reconstructed_data = clusters[nearest_indices]
    return nearest_indices, reconstructed_data


# Override the notebook k-means with the data-parallel variant from the VQDM repo.
@torch.no_grad()
def fit_kmeans(
    data: torch.Tensor,
    k: int,
    max_iter: int = 1000,
    check_every: int = 10,
    rtol: float = 1e-6,
    atol: float = 1e-8,
    greedy_init: bool = False,
    block_size_vals: int = 2**30,
    devices: Optional[List[torch.device]] = None,
):
    if devices is None:
        devices = [data.device]
    devices = [torch.device(d) for d in devices]
    if greedy_init:
        clusters0 = _kmeans_greedy_init(data, k)
    else:
        clusters0 = data[torch.randperm(data.shape[0], device=data.device)[:k], :].clone()

    block_size = max(1, block_size_vals // k)
    slices = _weighted_slices(len(data), [_device_vram_weight(device) for device in devices])
    devices = devices[:len(slices)]
    data_shards = [
        data[sl].to(devices[i], non_blocking=True)
        for i, sl in enumerate(slices)
    ]
    shard_size = max((len(shard) for shard in data_shards), default=0)
    nearest = [
        torch.empty(len(data_shards[i]), dtype=torch.int64, device=devices[i])
        for i in range(len(devices))
    ]
    clusters = [clusters0.to(device, non_blocking=True) for device in devices]

    for it in range(max_iter):
        for block_start in range(0, shard_size, block_size):
            for gi in range(len(devices)):
                block = data_shards[gi][block_start:block_start + block_size]
                if block.numel() == 0:
                    continue
                nearest[gi][block_start:block_start + block_size] = torch.addmm(
                    torch.bmm(clusters[gi][:, None, :], clusters[gi][:, :, None]).flatten(),
                    block,
                    clusters[gi].T,
                    beta=-0.5,
                ).argmax(1)

        if len(devices) == 1:
            new0 = clusters[0].clone().index_reduce_(
                dim=0,
                index=nearest[0],
                source=data_shards[0],
                reduce="mean",
                include_self=False,
            )
            new_clusters = [new0]
        else:
            cluster_sums = [
                torch.zeros_like(clusters[gi])
                .index_add(dim=0, index=nearest[gi], source=data_shards[gi])
                .to(devices[0], non_blocking=True)
                for gi in range(len(devices))
            ]
            cluster_counts = [
                torch.bincount(nearest[gi], minlength=k).to(devices[0], non_blocking=True)
                for gi in range(len(devices))
            ]
            for gi in range(1, len(devices)):
                cluster_sums[0] += cluster_sums[gi]
                cluster_counts[0] += cluster_counts[gi]
            new0 = cluster_sums[0] / cluster_counts[0].unsqueeze(1).clamp_min(1)
            new0 += (cluster_counts[0].unsqueeze(1) == 0) * clusters[0]
            new_clusters = [new0] + [new0.to(devices[gi], non_blocking=True) for gi in range(1, len(devices))]

        if it % check_every == 0 and torch.allclose(new_clusters[0], clusters[0], rtol=rtol, atol=atol):
            clusters = new_clusters
            break
        clusters = new_clusters

    for block_start in range(0, shard_size, block_size):
        for gi in range(len(devices)):
            block = data_shards[gi][block_start:block_start + block_size]
            if block.numel() == 0:
                continue
            nearest[gi][block_start:block_start + block_size] = torch.addmm(
                torch.bmm(clusters[gi][:, None, :], clusters[gi][:, :, None]).flatten(),
                block,
                clusters[gi].T,
                beta=-0.5,
            ).argmax(1)

    clusters_out = clusters[0]
    nearest_out = torch.cat([nearest[gi].to(devices[0], non_blocking=True) for gi in range(len(devices))], dim=0)
    reconstructed = clusters_out[nearest_out]
    return clusters_out, nearest_out, reconstructed


@torch.no_grad()
def find_nearest_cluster(
    data: torch.Tensor,
    clusters: torch.Tensor,
    block_size_vals: int = 2**30,
    devices: Optional[List[torch.device]] = None,
):
    if devices is None:
        devices = [data.device]
    devices = [torch.device(d) for d in devices]
    block_size = max(1, block_size_vals // len(clusters))
    slices = _weighted_slices(len(data), [_device_vram_weight(device) for device in devices])
    devices = devices[:len(slices)]
    data_shards = [
        data[sl].to(devices[i], non_blocking=True)
        for i, sl in enumerate(slices)
    ]
    shard_size = max((len(shard) for shard in data_shards), default=0)
    nearest = [
        torch.empty(len(data_shards[i]), dtype=torch.int64, device=devices[i])
        for i in range(len(devices))
    ]
    clusters_shards = [clusters.to(device, non_blocking=True) for device in devices]
    for block_start in range(0, shard_size, block_size):
        for gi in range(len(devices)):
            block = data_shards[gi][block_start:block_start + block_size]
            if block.numel() == 0:
                continue
            nearest[gi][block_start:block_start + block_size] = torch.addmm(
                torch.bmm(clusters_shards[gi][:, None, :], clusters_shards[gi][:, :, None]).flatten(),
                block,
                clusters_shards[gi].T,
                beta=-0.5,
            ).argmax(1)
    nearest_out = torch.cat([nearest[gi].to(devices[0], non_blocking=True) for gi in range(len(devices))], dim=0)
    clusters_out = clusters_shards[0]
    return nearest_out, clusters_out[nearest_out]

# %% cell 15
def fit_kmeans_1d(
    groupwise_data: torch.Tensor,
    k: int,
    max_iter: int = -1,
    offset_rate: float = 0.0,
    verbose: bool = False,
    initial_clusters: Optional[torch.Tensor] = None,
    rtol: float = 1e-6,
    atol: float = 1e-8,
):
    """
    groupwise_data: [num_groups, group_size]
    """
    assert groupwise_data.ndim == 2
    assert 0 <= offset_rate < 0.5

    sorted_data, groupwise_sort_indices = groupwise_data.sort(dim=1)
    groupwise_ranks_1based = groupwise_sort_indices.argsort(-1).add_(1)
    del groupwise_sort_indices

    sorted_cumsum = torch.cat([torch.zeros_like(sorted_data[:, :1]), sorted_data.cumsum(dim=1)], dim=1)

    if initial_clusters is not None:
        clusters = initial_clusters
    else:
        offset = int((sorted_data.shape[1] - 1) * offset_rate)
        init_indices = torch.linspace(offset, sorted_data.shape[1] - 1 - offset, k, dtype=torch.int64, device=sorted_data.device)
        clusters = sorted_data[:, init_indices]

    def _groupwise_find_border_indices(clusters, sorted_data):
        borders = (clusters[:, 1:] + clusters[:, :-1]) / 2
        col = clusters[:, :1]
        borders = torch.cat(
            [torch.full_like(col, float("-inf")), borders, torch.full_like(col, float("inf"))],
            dim=1
        )
        return torch.searchsorted(sorted_data, borders, side="left")

    i = 0
    while True:
        border_indices = _groupwise_find_border_indices(clusters, sorted_data)
        sum_by_cluster = torch.diff(sorted_cumsum.gather(1, border_indices), dim=1)
        count_by_cluster = torch.diff(border_indices, dim=1)

        new_clusters = torch.where(
            count_by_cluster > 0,
            sum_by_cluster / count_by_cluster,
            sorted_data.gather(1, border_indices[:, :-1].clamp_max(sorted_data.shape[1] - 1)),
        )

        if torch.allclose(new_clusters, clusters, rtol=rtol, atol=atol):
            if verbose:
                print(f"Early stopping after {i} iterations")
            clusters = new_clusters
            break

        clusters = new_clusters
        i += 1
        if max_iter > 0 and i >= max_iter:
            break

    border_indices = _groupwise_find_border_indices(clusters, sorted_data)
    groupwise_cluster_indices = torch.searchsorted(border_indices[:, 1:], groupwise_ranks_1based, side="left")
    groupwise_restored_data = clusters.gather(1, groupwise_cluster_indices)

    return clusters, groupwise_cluster_indices, groupwise_restored_data

# %% cell 16
@torch.no_grad()
def init_aq_kmeans(
    reference_weight: torch.Tensor,
    *,
    num_codebooks: int,
    out_group_size: int,
    in_group_size: int,
    codebook_size: int,
    groupwise_codebooks: bool,
    verbose: bool = False,
    use_faiss: bool = False,
    max_points_per_centroid: Optional[int] = None,
    max_iter: int = 1000,
    devices: Optional[List[torch.device]] = None,
):
    out_features, in_features = reference_weight.shape
    num_out_groups = out_features // out_group_size
    num_in_groups = in_features // in_group_size

    if groupwise_codebooks and num_out_groups != out_features:
        raise NotImplementedError("groupwise_codebooks with out_group_size != 1 not implemented")

    weight_residue = (
        reference_weight.reshape(num_out_groups, out_group_size, num_in_groups, in_group_size)
        .clone()
        .swapaxes(-3, -2)
        .reshape(num_out_groups * num_in_groups, out_group_size * in_group_size)
    )

    codebooks = []
    codes = []

    iterator = trange(num_codebooks, desc="initializing AQ with kmeans") if verbose else range(num_codebooks)

    for _ in iterator:
        if use_faiss:
            assert not groupwise_codebooks
            codebook_i, codes_i, reconstructed_i = fit_faiss_kmeans(
                weight_residue,
                k=codebook_size,
                max_iter=max_iter,
                gpu=(weight_residue.device.type == "cuda"),
                max_points_per_centroid=max_points_per_centroid,
                verbose=verbose,
            )
        else:
            chosen_ids = None
            if max_points_per_centroid is not None:
                assert not groupwise_codebooks
                chosen_ids = torch.randperm(weight_residue.shape[0], device=weight_residue.device)[
                    : max_points_per_centroid * codebook_size
                ]

            if not groupwise_codebooks:
                codebook_i, _, _ = fit_kmeans(
                    weight_residue if chosen_ids is None else weight_residue[chosen_ids, :],
                    k=codebook_size,
                    max_iter=max_iter,
                    devices=devices,
                )
                codes_i, reconstructed_i = find_nearest_cluster(weight_residue, codebook_i, devices=devices)
            else:
                codebook_i = []
                codes_i = []
                reconstructed_i = []
                assert chosen_ids is None

                for in_group_index in range(num_in_groups):
                    cb_ij, cd_ij, rec_ij = fit_kmeans(
                        weight_residue.reshape(num_out_groups, num_in_groups, -1)[:, in_group_index],
                        k=codebook_size,
                        max_iter=max_iter,
                        devices=devices,
                    )
                    codebook_i.append(cb_ij[None])
                    codes_i.append(cd_ij.reshape(num_out_groups, 1))
                    reconstructed_i.append(rec_ij.reshape(out_features, -1))

                codebook_i = torch.cat(codebook_i, dim=0)
                codes_i = torch.cat(codes_i, dim=1)
                reconstructed_i = (
                    torch.cat(reconstructed_i, dim=1)
                    .reshape(num_out_groups, out_group_size, num_in_groups, in_group_size)
                    .swapaxes(-3, -2)
                    .reshape(num_out_groups * num_in_groups, out_group_size * in_group_size)
                )

        codes_i = codes_i.reshape(num_out_groups, num_in_groups, 1)
        codebook_i = codebook_i.reshape(-1, 1, codebook_size, out_group_size, in_group_size)

        weight_residue -= reconstructed_i
        codes.append(codes_i)
        codebooks.append(codebook_i)

    codebooks = torch.cat(codebooks, dim=1)
    codes = torch.cat(codes, dim=-1)
    return codes, codebooks

# %% [markdown] cell 17
# <h1>Модули квантизации</h1>

# %% cell 18
class QuantizedLinear(nn.Module):
    def __init__(self, quantized_weight: "QuantizedWeight", bias: Optional[nn.Parameter]):
        super().__init__()
        self.out_features = quantized_weight.out_features
        self.in_features = quantized_weight.in_features
        self.quantized_weight = quantized_weight
        self.bias = bias

    def forward(self, x: torch.Tensor):
        w = self.quantized_weight().to(x.dtype)
        return F.linear(x, w, self.bias)


class QuantizedWeight(nn.Module):
    EPS = 1e-9

    def __init__(
        self,
        *,
        XTX: torch.Tensor,
        reference_weight: torch.Tensor,
        in_group_size: int,
        out_group_size: int,
        num_codebooks: int,
        nbits_per_codebook: int = 8,
        groupwise_codebooks: bool = False,
        input_channel_scales: bool = True,
        codebook_value_nbits: int = 16,
        codebook_value_num_groups: int = 1,
        scale_nbits: int = 0,
        code_dtype: torch.dtype = torch.int32,
        straight_through_gradient: Optional[bool] = None,
        **init_kwargs,
    ):
        super().__init__()

        self.out_features, self.in_features = reference_weight.shape
        assert self.in_features % in_group_size == 0
        assert self.out_features % out_group_size == 0

        if nbits_per_codebook > torch.iinfo(code_dtype).bits - is_signed(code_dtype):
            raise ValueError(
                f"Code dtype {code_dtype} cannot store {nbits_per_codebook} bits"
            )

        self.out_group_size = out_group_size
        self.in_group_size = in_group_size
        self.num_codebooks = num_codebooks
        self.nbits_per_codebook = nbits_per_codebook
        self.codebook_size = 2 ** nbits_per_codebook
        self.codebook_value_nbits = codebook_value_nbits
        self.codebook_value_num_groups = codebook_value_num_groups
        self.codebook_value_clusters = None

        self.scales = None
        self.scales_clusters = None
        self.scales_indices = None

        if straight_through_gradient is None and scale_nbits > 0:
            straight_through_gradient = scale_nbits >= 6
        self.straight_through_gradient = straight_through_gradient
        self.scale_nbits = scale_nbits

        with torch.no_grad():
            weight_groupwise = reference_weight.reshape(
                self.out_features // out_group_size,
                out_group_size,
                self.in_features // in_group_size,
                in_group_size,
            ).swapaxes(1, 2)
            # [num_out_groups, num_in_groups, out_group_size, in_group_size]

            if scale_nbits > 0:
                scales = weight_groupwise.norm(dim=(2, 3), keepdim=True) + self.EPS
            else:
                scales = weight_groupwise.flatten(1, -1).norm(dim=-1).view(-1, 1, 1, 1) + self.EPS

            self.scales_are_lossless = (
                scale_nbits == 0 or scale_nbits >= 16 or (2 ** scale_nbits >= scales.shape[1])
            )

            if self.scales_are_lossless or self.straight_through_gradient:
                self.scales = nn.Parameter(scales, requires_grad=True)
            else:
                scales_clusters, scales_indices, _ = fit_kmeans_1d(scales.flatten(1, -1), k=2 ** scale_nbits)
                self.scales_clusters = nn.Parameter(scales_clusters, requires_grad=True)
                self.scales_indices = nn.Parameter(scales_indices, requires_grad=False)

            weight_for_init = (weight_groupwise / scales).swapaxes(1, 2).reshape_as(reference_weight)
            del weight_groupwise

            self.input_scales = None
            if input_channel_scales:
                self.input_scales = nn.Parameter(
                    weight_for_init.square().sum(dim=0).sqrt() + self.EPS,
                    requires_grad=True,
                )
                weight_for_init = weight_for_init / self.input_scales

        codes, codebooks = init_aq_kmeans(
            reference_weight=weight_for_init,
            num_codebooks=num_codebooks,
            out_group_size=out_group_size,
            in_group_size=in_group_size,
            codebook_size=self.codebook_size,
            groupwise_codebooks=groupwise_codebooks,
            **init_kwargs,
        )

        assert codebooks.ndim == 5
        assert codebooks.shape[0] in (1, self.in_features // self.in_group_size)

        self.codebooks = nn.Parameter(codebooks, requires_grad=True)
        # [num_in_groups or 1, num_codebooks, codebook_size, out_group_size, in_group_size]

        self.codes = nn.Parameter(
            codes.to(code_dtype),
            requires_grad=False,
        )
        # [num_out_groups, num_in_groups, num_codebooks]

        if globals().get("VQDM_STORE_XTX_IN_QWEIGHT", False):
            self.register_buffer("XTX_buffer", XTX.detach().clone())
        else:
            self.XTX_buffer = None

    def get_codes(self) -> torch.Tensor:
        if torch.iinfo(self.codes.dtype).bits < 32:
            return self.codes.to(torch.int32)
        return self.codes

    def set_codes(self, new_codes: torch.Tensor, selection: Union[slice, ellipsis, torch.Tensor] = ...):
        self.codes[selection] = new_codes.to(dtype=self.codes.dtype, device=self.codes.device)

    def get_codebooks(self) -> torch.Tensor:
        if self.codebook_value_nbits >= 16:
            return self.codebooks

        if 0 < self.codebook_value_nbits < 16:
            with torch.no_grad():
                codebooks_dimshuffle = (
                    self.codebooks.reshape(
                        self.codebooks.shape[0],
                        self.num_codebooks,
                        self.codebook_value_num_groups,
                        self.codebook_size // self.codebook_value_num_groups,
                        self.out_group_size,
                        self.in_group_size,
                    )
                    .permute(2, 4, 5, 0, 1, 3)
                    .flatten(0, -3)
                    .flatten(1)
                )
                self.codebook_value_clusters, _, reconstructed = fit_kmeans_1d(
                    codebooks_dimshuffle,
                    k=2 ** self.codebook_value_nbits,
                    initial_clusters=self.codebook_value_clusters,
                )
                reconstructed = (
                    reconstructed.view(
                        self.codebook_value_num_groups,
                        self.out_group_size,
                        self.in_group_size,
                        self.codebooks.shape[0],
                        self.num_codebooks,
                        self.codebook_size // self.codebook_value_num_groups,
                    )
                    .permute(3, 4, 0, 5, 1, 2)
                    .reshape_as(self.codebooks)
                )
            if torch.is_grad_enabled():
                reconstructed = reconstructed + (self.codebooks - self.codebooks.detach())
            return reconstructed

        raise NotImplementedError(f"{self.codebook_value_nbits}-bit codebook values are not supported")

    def get_scales(self) -> torch.Tensor:
        if self.scale_nbits == 0 or self.scales_are_lossless:
            return self.scales

        if self.straight_through_gradient:
            with torch.no_grad():
                self.scales_clusters, _, dequantized_scales = fit_kmeans_1d(
                    self.scales.flatten(1, -1),
                    k=2 ** self.scale_nbits,
                    initial_clusters=self.scales_clusters,
                )
                dequantized_scales = dequantized_scales.reshape_as(self.scales)
            if torch.is_grad_enabled() and self.scales.requires_grad:
                dequantized_scales = dequantized_scales + (self.scales - self.scales.detach())
            return dequantized_scales

        return self.scales_clusters.gather(1, self.scales_indices)[:, :, None, None]

    def forward(self, selection: Union[slice, ellipsis, torch.Tensor] = ...):
        weight = _dequantize_weight(
            self.get_codes()[selection],
            self.get_codebooks(),
            self.get_scales()[selection],
        )
        if self.input_scales is not None:
            weight = weight * self.input_scales
        return weight

    @torch.no_grad()
    def beam_search_update_codes_(
        self,
        XTX: torch.Tensor,
        reference_weight: torch.Tensor,
        *,
        selection: Union[slice, ellipsis, torch.LongTensor] = ...,
        **kwargs,
    ):
        if self.input_scales is not None:
            XTX = XTX.mul(self.input_scales[None, :]).mul_(self.input_scales[:, None])
            reference_weight = reference_weight.div(self.input_scales[None, :])

        new_codes = beam_search_optimal_codes(
            XTX=XTX,
            reference_weight=reference_weight,
            codebooks=self.get_codebooks(),
            prev_codes=self.get_codes()[selection],
            scales=self.get_scales()[selection],
            **kwargs,
        )
        self.set_codes(new_codes, selection)
        return self.get_codes()[selection]

    def estimate_nbits_per_parameter(self) -> float:
        num_parameters = self.out_features * self.in_features
        group_size = self.out_group_size * self.in_group_size
        num_out_groups = self.out_features // self.out_group_size
        num_in_groups = self.in_features // self.in_group_size

        matrix_store = num_parameters // group_size * self.num_codebooks * self.nbits_per_codebook

        codebooks_store = self.num_codebooks * self.codebook_size * group_size * self.codebook_value_nbits
        if self.codebook_value_nbits < 16:
            codebooks_store += 2 ** self.codebook_value_nbits * self.codebook_value_num_groups * group_size * 16

        if self.scale_nbits >= 16 or 2 ** self.scale_nbits >= num_in_groups:
            scale_store = self.scale_nbits * num_out_groups * num_in_groups
        elif 0 < self.scale_nbits < 16:
            scale_store = self.scale_nbits * num_out_groups * num_in_groups
            scale_store += num_out_groups * 2 ** self.scale_nbits * 16
        elif self.scale_nbits == 0:
            scale_store = num_out_groups * 16
        else:
            raise RuntimeError("Unexpected scale_nbits")

        return (matrix_store + codebooks_store + scale_store) / num_parameters

# %% cell 19
class QuantizedConv2D(nn.Module):
    def __init__(
        self,
        quantized_weight: "QuantizedWeight",
        bias: Optional[nn.Parameter],
        in_channels: int,
        out_channels: int,
        kernel_size: Tuple[int, ...],
        stride: Tuple[int, ...],
        padding: Tuple[int, ...],
        dilation: Tuple[int, ...],
        group_channels: bool,
    ):
        super().__init__()
        self.quantized_weight = quantized_weight
        self.bias = bias

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.group_channels = group_channels

    @classmethod
    def from_conv(
        cls,
        quantized_weight: "QuantizedWeight",
        conv: nn.Conv2d,
        group_channels: bool = True,
    ):
        assert conv.groups == 1, "groups != 1 not supported in this minimal version"
        return cls(
            quantized_weight=quantized_weight,
            bias=conv.bias,
            in_channels=conv.in_channels,
            out_channels=conv.out_channels,
            kernel_size=conv.kernel_size,
            stride=conv.stride,
            padding=conv.padding,
            dilation=conv.dilation,
            group_channels=group_channels,
        )

    def forward(self, x: torch.Tensor):
        if self.group_channels:
            w = (
                self.quantized_weight()
                .reshape(self.out_channels, *self.kernel_size, self.in_channels)
                .permute(0, 3, 1, 2)
                .contiguous()
            )
        else:
            w = self.quantized_weight().reshape(
                self.out_channels, self.in_channels, *self.kernel_size
            ).contiguous()

        return F.conv2d(
            x,
            w.to(x.dtype),
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
        )

# %% [markdown] cell 20
# <h1>Деквантизация и подсчет ошибки для beam-search </h1>

# %% cell 21
@maybe_script
def _dequantize_weight(
    codes: torch.Tensor,
    codebooks: torch.Tensor,
    scales: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    codes: [..., num_out_groups, num_in_groups, num_codebooks]
    codebooks: [num_in_groups or 1, num_codebooks, codebook_size, out_group_size, in_group_size]
    """
    assert codebooks.ndim == 5

    num_out_groups, num_in_groups, num_codebooks = codes.shape[-3:]
    num_in_groups_or_1, num_codebooks, codebook_size, out_group_size, in_group_size = codebooks.shape

    out_features = num_out_groups * out_group_size
    in_features = num_in_groups * in_group_size

    codebook_offsets = torch.arange(
        0,
        num_in_groups_or_1 * num_codebooks * codebook_size,
        codebook_size,
        device=codes.device,
    ).reshape(num_in_groups_or_1, num_codebooks)

    reconstructed_flat = F.embedding_bag(
        (codes + codebook_offsets).flatten(0, -2),
        codebooks.flatten(0, 2).flatten(-2, -1),
        mode="sum",
    )

    reconstructed_groupwise = reconstructed_flat.view(
        list(codes.shape[:-3]) + [num_out_groups, num_in_groups, out_group_size, in_group_size]
    )

    if scales is not None:
        reconstructed_groupwise = reconstructed_groupwise * scales

    return reconstructed_groupwise.swapaxes(-3, -2).reshape(
        list(codes.shape[:-3]) + [out_features, in_features]
    )

# %% cell 22
@maybe_script
def _channelwise_squared_error(
    XTX: torch.Tensor,
    weight: torch.Tensor,
    reference_weight: torch.Tensor,
):
    XW_norm_square = torch.matmul(
        weight[..., :, None, :],
        (weight @ XTX)[..., :, :, None]
    ).flatten(-3)

    XWreference_norm_square = torch.bmm(
        reference_weight[:, None, :],
        (reference_weight @ XTX)[:, :, None]
    ).flatten(-3)

    dot_product = torch.matmul(
        (reference_weight @ XTX)[:, None, :],
        weight[..., :, :, None]
    ).flatten(-3)

    return XW_norm_square - 2 * dot_product + XWreference_norm_square

# %% cell 23
@maybe_script
def _beam_search_squared_errors(
    XTX: torch.Tensor,
    reference_weight: torch.Tensor,
    codebooks: torch.Tensor,
    scales: Optional[torch.Tensor],
    beam_losses: torch.Tensor,
    beam_codes: torch.Tensor,
    beam_weights: torch.Tensor,
    input_group_index: int,
    codebook_index: int,
    k_best: int,
    sparsity_regularizer: float,
):
    num_codebooks, codebook_size, out_group_size, in_group_size = codebooks.shape
    beam_size, num_out_groups, num_in_groups, num_codebooks = beam_codes.shape
    out_features = num_out_groups * out_group_size

    input_group_slice = slice(
        input_group_index * in_group_size,
        (input_group_index + 1) * in_group_size,
    )

    prev_codes_part = beam_codes[:, :, input_group_index, codebook_index]

    if scales is not None:
        scales_part = scales[:, input_group_index % scales.shape[1], :, :]
    else:
        scales_part = torch.empty(0, device=XTX.device, dtype=XTX.dtype)

    prev_part_dequantized = F.embedding(
        prev_codes_part,
        codebooks[codebook_index].flatten(-2, -1)
    ).view(beam_size, out_features, in_group_size)

    prev_weight_part = prev_part_dequantized
    if scales is not None:
        prev_weight_part = (
            prev_weight_part.view(beam_size, num_out_groups, out_group_size, in_group_size)
            .mul(scales_part)
            .view(beam_size, out_features, in_group_size)
        )

    cand_weights = codebooks[codebook_index]

    delta_weight_without_part = reference_weight - beam_weights
    delta_weight_without_part[:, :, input_group_slice] += prev_weight_part

    dWTXTXg = delta_weight_without_part @ XTX[..., input_group_slice]

    XnewBkC_norms_sq = torch.bmm(
        (cand_weights.flatten(0, 1) @ XTX[input_group_slice, input_group_slice]).view(
            codebook_size, 1, out_group_size * in_group_size
        ),
        cand_weights.view(codebook_size, out_group_size * in_group_size, 1),
    ).reshape(codebook_size, 1)

    if scales is not None:
        XnewBkC_norms_sq = XnewBkC_norms_sq.mul(scales_part.square().reshape(1, num_out_groups))

    best_losses = torch.empty(
        (beam_size, k_best, num_out_groups),
        dtype=XTX.dtype,
        device=XTX.device,
    )
    best_indices = torch.empty(
        (beam_size, k_best, num_out_groups),
        dtype=torch.int64,
        device=XTX.device,
    )

    for beam_id in range(beam_size):
        dot_products = (
            torch.einsum(
                "mg,og->mo",
                cand_weights.reshape(codebook_size, out_group_size * in_group_size),
                dWTXTXg[beam_id].view(num_out_groups, out_group_size * in_group_size),
            )
            .sub_(
                torch.einsum(
                    "og,og->o",
                    prev_part_dequantized[beam_id].reshape(num_out_groups, out_group_size * in_group_size),
                    dWTXTXg[beam_id].view(num_out_groups, out_group_size * in_group_size),
                ).view(1, num_out_groups)
            )
            .view(codebook_size, num_out_groups)
        )

        if scales is not None:
            dot_products = dot_products.mul_(scales_part.reshape(1, num_out_groups))

        XoldBkC_norms_sq = torch.bmm(
            (prev_weight_part[beam_id] @ XTX[input_group_slice, input_group_slice]).view(
                num_out_groups, 1, out_group_size * in_group_size
            ),
            prev_weight_part[beam_id].view(num_out_groups, out_group_size * in_group_size, 1),
        ).reshape(1, num_out_groups)

        candidate_squared_errors = (
            beam_losses[beam_id, None, :] - 2 * dot_products + XnewBkC_norms_sq - XoldBkC_norms_sq
        )

        if sparsity_regularizer != 0:
            candidate_squared_errors += sparsity_regularizer * (prev_codes_part[beam_id] == 0).to(XTX.dtype)[None, :]
            candidate_squared_errors[0, :] -= sparsity_regularizer

        best_beam_squared_errors, best_beam_indices = torch.topk(
            candidate_squared_errors,
            k_best,
            dim=0,
            largest=False,
            sorted=False,
        )

        best_losses[beam_id] = best_beam_squared_errors
        best_indices[beam_id] = best_beam_indices

    return best_losses, best_indices

# %% cell 24
@maybe_script
def _beam_search_select_best(
    beam_codes: torch.Tensor,
    beam_weights: torch.Tensor,
    codebooks: torch.Tensor,
    scales: Optional[torch.Tensor],
    input_group_index: int,
    codebook_index: int,
    best_losses: torch.Tensor,
    best_indices: torch.Tensor,
    beam_size: int,
):
    dtype = best_losses.dtype
    device = best_losses.device

    _prev_beam_size, k_best, num_out_groups = best_losses.shape
    _prev_beam_size, out_features, in_features = beam_weights.shape
    _prev_beam_size, num_out_groups, num_in_groups, num_codebooks = beam_codes.shape

    flat_best = best_losses.flatten(0, 1).topk(dim=0, k=beam_size, largest=False)
    best_hypo_source_ids = flat_best.indices // k_best
    arange_out_groups = torch.arange(num_out_groups, device=device)

    best_hypo_codes = best_indices.flatten(0, 1)[flat_best.indices, arange_out_groups].reshape(
        beam_size, num_out_groups
    )

    new_beam_codes = torch.full(
        (beam_size, num_out_groups, num_in_groups, num_codebooks),
        fill_value=-1,
        dtype=beam_codes.dtype,
        device=device,
    )
    new_beam_weights = torch.empty(
        beam_size,
        out_features,
        in_features,
        dtype=dtype,
        device=device,
    )

    for beam_index in range(beam_size):
        new_beam_codes[beam_index, :, ...] = beam_codes[
            best_hypo_source_ids[beam_index, :],
            arange_out_groups,
            ...
        ]
        new_beam_codes[beam_index, :, input_group_index, codebook_index] = best_hypo_codes[beam_index, :]
        new_beam_weights[beam_index, :, :] = _dequantize_weight(
            new_beam_codes[beam_index, ...],
            codebooks,
            scales,
        )

    return new_beam_codes, new_beam_weights, flat_best.values

# %% [markdown] cell 25
# <h1>Beam-search</h1>

# %% cell 26
@torch.inference_mode()
def beam_search_optimal_codes(
    *,
    XTX: torch.Tensor,
    reference_weight: torch.Tensor,
    codebooks: torch.Tensor,
    prev_codes: torch.IntTensor,
    scales: Optional[torch.Tensor],
    beam_size: int,
    dim_rng: Optional[random.Random] = None,
    sparsity_regularizer: float = 0.0,
    verbose: bool = False,
):
    start_time = time.perf_counter()

    assert codebooks.ndim == 5
    num_out_groups, num_in_groups, num_codebooks = prev_codes.shape
    _, num_codebooks, codebook_size, out_group_size, in_group_size = codebooks.shape

    in_features = num_in_groups * in_group_size
    out_features = num_out_groups * out_group_size
    assert reference_weight.shape == (out_features, in_features)

    prev_weight = _dequantize_weight(prev_codes, codebooks, scales)

    beam_codes = prev_codes.unsqueeze(0)
    beam_weights = prev_weight.unsqueeze(0)

    beam_losses = (
        _channelwise_squared_error(XTX, prev_weight, reference_weight)
        .reshape(1, num_out_groups, out_group_size)
        .sum(-1)
    )

    if sparsity_regularizer != 0:
        beam_losses = beam_losses - sparsity_regularizer * (prev_codes == 0).sum(dim=(-1, -2))[None, :]

    def _make_range(n: int):
        seq = list(range(n))
        if dim_rng is not None:
            dim_rng.shuffle(seq)
        return seq

    input_group_iter = _make_range(num_in_groups)
    if verbose:
        input_group_iter = tqdm(input_group_iter, desc="beam search groups", leave=False, unit="group")

    for input_group_index in input_group_iter:
        codebook_iter = _make_range(num_codebooks)
        if verbose:
            codebook_iter = tqdm(codebook_iter, desc=f"group {input_group_index} codebooks", leave=False, unit="codebook")

        for codebook_index in codebook_iter:
            best_losses, best_indices = _beam_search_squared_errors(
                XTX=XTX,
                reference_weight=reference_weight,
                codebooks=codebooks[input_group_index % codebooks.shape[0]],
                scales=scales,
                beam_losses=beam_losses,
                beam_codes=beam_codes,
                beam_weights=beam_weights,
                input_group_index=input_group_index,
                codebook_index=codebook_index,
                k_best=beam_size,
                sparsity_regularizer=sparsity_regularizer,
            )

            beam_codes, beam_weights, beam_losses = _beam_search_select_best(
                beam_codes=beam_codes,
                beam_weights=beam_weights,
                codebooks=codebooks,
                scales=scales,
                input_group_index=input_group_index,
                codebook_index=codebook_index,
                best_losses=best_losses,
                best_indices=best_indices,
                beam_size=beam_size,
            )

    if verbose:
        best_loss = beam_losses.min(0).values.sum().item() / out_features
        elapsed = time.perf_counter() - start_time
        print(
            f"[beam_search] num_in_groups={num_in_groups}, num_codebooks={num_codebooks}, "
            f"iters={num_in_groups * num_codebooks}, best_loss={best_loss:.10f}, t={elapsed:.3f}s"
        )

    return beam_codes[0]

# %% [markdown] cell 27
# <h1>AQ</h1>

# %% cell 28
class AQEngine(nn.Module):
    def __init__(
        self,
        layer: Union[nn.Linear, nn.Conv2d],
        accum_dtype: torch.dtype = torch.float64,
        group_channels: bool = True,
        xtx_device: str = "cpu",
    ):
        super().__init__()
        assert isinstance(layer, (nn.Linear, nn.Conv2d))
        self.layer = layer
        self.device = layer.weight.device
        self.group_channels = group_channels
        self.xtx_device = torch.device(xtx_device)

        cols = self.reference_weight.shape[1]

        self.register_buffer(
            "XTX",
            torch.zeros(
                (cols, cols),
                dtype=accum_dtype,
                device=self.xtx_device,
            )
        )
        self.quantized_weight: Optional[QuantizedWeight] = None
        self.nsamples = 0

    @property
    @torch.no_grad()
    def reference_weight(self) -> torch.Tensor:
        w = self.layer.weight.detach().to(dtype=torch.float32)

        if isinstance(self.layer, nn.Conv2d):
            if self.group_channels:
                w = w.permute(0, 2, 3, 1).contiguous()
            return w.flatten(1)

        return w.flatten(1)

    @torch.no_grad()
    def add_batch(self, inp: torch.Tensor):
        if isinstance(self.layer, nn.Conv2d):
            unfold = nn.Unfold(
                kernel_size=self.layer.kernel_size,
                dilation=self.layer.dilation,
                padding=self.layer.padding,
                stride=self.layer.stride,
            )
            inp = unfold(inp)  # [B, C*k*k, L]
            if self.group_channels:
                B, _, L = inp.shape
                Cin = self.layer.in_channels
                Kh, Kw = self.layer.kernel_size
                inp = inp.view(B, Cin, Kh, Kw, L).permute(0, 2, 3, 1, 4).contiguous()
                inp = inp.view(B, Kh * Kw * Cin, L)
            inp = inp.permute(1, 0, 2).flatten(1)  # [cols, B*L]
        else:
            if inp.ndim == 3:
                inp = inp.reshape(-1, inp.shape[-1])
            elif inp.ndim == 2:
                pass
            else:
                raise ValueError(f"Unexpected input shape: {tuple(inp.shape)}")
            inp = inp.t()

        new_samples = inp.shape[1]
        self.XTX *= self.nsamples / (self.nsamples + new_samples)
        self.nsamples += new_samples

        inp = math.sqrt(1.0 / self.nsamples) * inp.to(self.XTX.device, dtype=self.XTX.dtype)
        self.XTX += inp.matmul(inp.t())

    def _compute_mse(self, selection: Union[slice, ellipsis] = ...) -> torch.Tensor:
        assert self.quantized_weight is not None
    
        quantized_weight = self.quantized_weight(selection)
    
        if isinstance(selection, ellipsis):
            reference_weight = self.reference_weight.detach().to(
                device=quantized_weight.device,
                dtype=quantized_weight.dtype,
            )
        else:
            assert isinstance(selection, slice)
            out_channel_selection = slice(
                selection.start * self.quantized_weight.out_group_size,
                selection.stop * self.quantized_weight.out_group_size,
            )
            reference_weight = self.reference_weight.detach()[out_channel_selection].to(
                device=quantized_weight.device,
                dtype=quantized_weight.dtype,
            )
    
        xtx = self.XTX.to(
            device=quantized_weight.device,
            dtype=quantized_weight.dtype,
        )
    
        delta_weight = (quantized_weight - reference_weight).to(xtx.dtype)
        return (delta_weight @ xtx).flatten() @ delta_weight.flatten() / self.quantized_weight.out_features

    @torch.enable_grad()
    def quantize(self, cfg: AQConfig, verbose: bool = True) -> QuantizedWeight:
        self.quantized_weight = QuantizedWeight(
            XTX=self.XTX.to(device=self.device, dtype=torch.float32),
            reference_weight=self.reference_weight.to(self.device),
            out_group_size=cfg.out_group_size,
            in_group_size=cfg.in_group_size,
            num_codebooks=cfg.num_codebooks,
            nbits_per_codebook=cfg.nbits_per_codebook,
            groupwise_codebooks=cfg.groupwise_codebooks,
            codebook_value_nbits=cfg.codebook_value_nbits,
            codebook_value_num_groups=cfg.codebook_value_num_groups,
            scale_nbits=cfg.scale_nbits,
            input_channel_scales=cfg.input_channel_scales,
            max_iter=cfg.init_max_iter,
            max_points_per_centroid=cfg.init_max_points_per_centroid,
            use_faiss=cfg.use_faiss,
            code_dtype=cfg.code_dtype,
            devices=list(cfg.devices) if cfg.devices is not None else [self.device],
        )

        differentiable_parameters = nn.ParameterDict(
            {name: p for name, p in self.quantized_weight.named_parameters() if p.requires_grad}
        )
        devices = [torch.device(d) for d in (cfg.devices or [self.device])]
        if devices[0] != self.device:
            devices = [self.device] + [d for d in devices if d != self.device]

        opt = torch.optim.Adam(
            differentiable_parameters.values(),
            lr=cfg.lr,
            betas=(0.0, 0.95),
            amsgrad=True,
        )
        replicas = None
        if len(devices) > 1:
            log_event(f"[AQEngine] multi-GPU AQ enabled on {[str(d) for d in devices]}")
            replicas = torch.nn.parallel.replicate(self, devices)
            replicas[0] = self

        previous_best_loss = float("inf")

        epoch_iter = range(cfg.max_epochs)
        if verbose:
            epoch_iter = tqdm(epoch_iter, desc="AQ epochs", leave=False, unit="epoch")

        for epoch in epoch_iter:
            epoch_start = time.perf_counter()
            step_losses = []

            step_iter = range(cfg.steps_per_epoch)
            if verbose:
                step_iter = tqdm(step_iter, desc=f"AQ epoch {epoch} steps", leave=False, unit="step")

            for step in step_iter:
                if replicas is None:
                    loss = self._compute_mse()
                else:
                    loss = self._compute_mse_parallel(devices, replicas, differentiable_parameters)

                if not torch.isfinite(loss).item():
                    raise ValueError(f"Quantization loss is not finite: {loss.item()}")

                if step == 0 and cfg.relative_mse_tolerance is not None:
                    if loss.item() / previous_best_loss > (1.0 - cfg.relative_mse_tolerance):
                        if verbose:
                            print(f"[AQEngine] Early stop at epoch={epoch}, loss={loss.item():.10f}")
                        return self.quantized_weight
                    previous_best_loss = min(previous_best_loss, loss.item())

                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                step_losses.append(loss.item())
                if verbose and hasattr(step_iter, "set_postfix"):
                    step_iter.set_postfix(loss=f"{loss.item():.4g}")

            seed = cfg.seed + 100003 * epoch
            self.beam_search_update_codes_(
                devices=devices,
                replicas=replicas,
                parameters_to_replicate=differentiable_parameters,
                seed=seed,
                beam_size=cfg.beam_size,
                verbose=verbose,
            )

            if verbose:
                elapsed = time.perf_counter() - epoch_start
                if hasattr(epoch_iter, "set_postfix"):
                    epoch_iter.set_postfix(loss=f"{min(step_losses):.4g}")
                print(
                    f"[AQEngine] epoch={epoch:02d} "
                    f"first={step_losses[0]:.10f} "
                    f"min={min(step_losses):.10f} "
                    f"last={step_losses[-1]:.10f} "
                    f"t={elapsed:.2f}s"
                )

        return self.quantized_weight

    def _replace_and_compute_mse(self, params_to_replace: nn.ParameterDict, selection: slice) -> torch.Tensor:
        for param_name, param_value in params_to_replace.items():
            replace_parameter_(self.quantized_weight, param_name, param_value)
        return self._compute_mse(selection)

    def _compute_mse_parallel(
        self,
        devices: Sequence[torch.device],
        replicas: Sequence["AQEngine"],
        parameters_to_replicate: nn.ParameterDict,
    ) -> torch.Tensor:
        replicated_parameters = torch.nn.parallel.replicate(parameters_to_replicate, devices, detach=False)
        num_output_groups = self.quantized_weight.out_features // self.quantized_weight.out_group_size
        active_slices = _weighted_slices(num_output_groups, [_device_vram_weight(device) for device in devices])
        devices = list(devices)[:len(active_slices)]
        replicas = list(replicas)[:len(active_slices)]
        funcs = [replica._replace_and_compute_mse for replica in replicas]
        inputs = [(dict(), active_slices[0])]
        for i in range(1, len(devices)):
            inputs.append((replicated_parameters[i], active_slices[i]))
        mse_parts = torch.nn.parallel.parallel_apply(funcs, inputs, devices=devices)
        return Gather.apply(devices[0], 0, *(mse.view(1) for mse in mse_parts)).sum()

    def _replace_and_beam_search(
        self,
        params_to_replace: nn.ParameterDict,
        selection: slice,
        **kwargs,
    ) -> torch.Tensor:
        dtype = self.quantized_weight.codebooks.dtype
        for param_name, param_value in params_to_replace.items():
            replace_parameter_(self.quantized_weight, param_name, param_value)
        out_channel_selection = slice(
            selection.start * self.quantized_weight.out_group_size,
            selection.stop * self.quantized_weight.out_group_size,
        )
        reference_weight = self.reference_weight.detach()[out_channel_selection].to(dtype)
        return self.quantized_weight.beam_search_update_codes_(
            self.XTX.to(dtype),
            reference_weight,
            selection=selection,
            **kwargs,
        ).clone()

    @torch.no_grad()
    def beam_search_update_codes_(
        self,
        devices: Sequence[torch.device],
        replicas: Optional[Sequence["AQEngine"]],
        parameters_to_replicate: nn.ParameterDict,
        seed: Optional[int] = None,
        beam_size: int = 1,
        verbose: bool = False,
    ):
        dtype = self.quantized_weight.codebooks.dtype
        if len(devices) == 1:
            self.quantized_weight.beam_search_update_codes_(
                self.XTX.to(self.device, dtype=dtype),
                self.reference_weight.detach().to(self.device, dtype=dtype),
                dim_rng=random.Random(seed),
                beam_size=beam_size,
                verbose=verbose,
            )
            return

        assert replicas is not None and replicas[0] is self
        replicated_parameters = torch.nn.parallel.replicate(parameters_to_replicate, devices)
        num_output_groups = self.quantized_weight.out_features // self.quantized_weight.out_group_size
        active_slices = _weighted_slices(num_output_groups, [_device_vram_weight(device) for device in devices])
        devices = list(devices)[:len(active_slices)]
        replicas = list(replicas)[:len(active_slices)]
        funcs = [replica._replace_and_beam_search for replica in replicas]
        inputs = [(dict(), active_slices[0])]
        for i in range(1, len(devices)):
            inputs.append((replicated_parameters[i], active_slices[i]))
        kwargs_by_replica = [
            dict(dim_rng=random.Random(seed), beam_size=beam_size, verbose=verbose)
            for _ in range(len(devices))
        ]
        new_code_parts = torch.nn.parallel.parallel_apply(
            funcs,
            inputs,
            kwargs_by_replica,
            devices=devices,
        )
        gathered = Gather.apply(devices[0], 0, *new_code_parts)
        for device, replica in zip(devices, replicas):
            replica.quantized_weight.set_codes(gathered.to(device))


def replace_parameter_(module: nn.Module, name: str, new_value: torch.Tensor):
    if name in module._parameters:
        module._parameters[name] = new_value
    else:
        setattr(module, name, new_value)

# %% [markdown] cell 29
# <h1>Сбор калибровочных данных</h1>

# %% cell 30
class _LayerWrapperThatAccumulatesXTX(nn.Module):
    def __init__(self, layer: nn.Module, aq_handler: AQEngine):
        super().__init__()
        self.wrapped_layer = layer
        self.aq_handler = aq_handler

    def forward(self, input, *args, **kwargs):
        self.aq_handler.add_batch(input)
        return self.wrapped_layer(input, *args, **kwargs)

# %% cell 31
def flatten_linear_input(x: torch.Tensor) -> torch.Tensor:
    # x: [..., C]
    if x.ndim == 2:
        return x
    return x.reshape(-1, x.shape[-1])


@torch.no_grad()
def collect_midblock_xtx_streaming(
    pipe,
    calibration_prompts,
    layer_names=None,
    xtx_batch_size=4,
    num_inference_steps=50,
    guidance_scale=5.0,
    height=None,
    width=None,
    collect_device="cuda",
    xtx_store_device="cpu",
    accum_dtype=torch.float64,
    group_channels=True,
    show_progress=True,
    generator_seed: Optional[int] = None,
):
    collect_dtype = globals().get("runtime_dtype", torch.float16) if str(collect_device).startswith("cuda") else torch.float32
    pipe = force_pipeline_device_dtype(pipe, collect_device, collect_dtype)
    pipe.set_progress_bar_config(disable=True)
    xtx_store_device = torch.device(xtx_store_device)
    mid_block = pipe.unet.mid_block

    gen_device = collect_device if str(collect_device).startswith("cuda") else "cpu"

    stats = {}
    hooks = []

    for name, module in mid_block.named_modules():
        full_name = f"mid_block.{name}"
        if not isinstance(module, (nn.Linear, nn.Conv2d)):
            continue
        if layer_names is not None and full_name not in layer_names:
            continue

        if isinstance(module, nn.Linear):
            cols = module.in_features
        else:
            cols = module.in_channels * module.kernel_size[0] * module.kernel_size[1]

        stats[full_name] = {
            "xtx": torch.zeros((cols, cols), dtype=accum_dtype, device=xtx_store_device),
            "n_samples": 0,
            "layer": module,
        }

    def make_hook(layer_full_name):
        def hook(module, inputs, output):
            x = inputs[0].detach()

            if isinstance(module, nn.Linear):
                if x.ndim == 3:
                    x = x.reshape(-1, x.shape[-1])
                elif x.ndim == 2:
                    pass
                else:
                    raise ValueError(
                        f"Unexpected Linear input shape for {layer_full_name}: {tuple(x.shape)}"
                    )
                x = x.to(xtx_store_device, dtype=torch.float32)  # [N, C]
                stats[layer_full_name]["xtx"] += x.t().to(accum_dtype) @ x.to(accum_dtype)
                stats[layer_full_name]["n_samples"] += x.shape[0]

            elif isinstance(module, nn.Conv2d):
                unfold = nn.Unfold(
                    kernel_size=module.kernel_size,
                    dilation=module.dilation,
                    padding=module.padding,
                    stride=module.stride,
                )
                x = unfold(x)  # [B, C*k*k, L]

                if group_channels:
                    # match AQEngine.reference_weight for conv:
                    # weight permute(0,2,3,1).flatten(1), so columns are [k1, k2, c]
                    B, CK2, L = x.shape
                    Cin = module.in_channels
                    Kh, Kw = module.kernel_size
                    x = x.view(B, Cin, Kh, Kw, L).permute(0, 2, 3, 1, 4).contiguous()
                    x = x.view(B, Kh * Kw * Cin, L)  # [B, k*k*C, L]
                # else keep default unfold order [B, C*k*k, L]

                x = x.permute(1, 0, 2).reshape(x.shape[1], -1).t()  # [B*L, cols]
                x = x.to(xtx_store_device, dtype=torch.float32)

                stats[layer_full_name]["xtx"] += x.t().to(accum_dtype) @ x.to(accum_dtype)
                stats[layer_full_name]["n_samples"] += x.shape[0]

            else:
                raise TypeError(type(module))

        return hook

    for name, module in mid_block.named_modules():
        full_name = f"mid_block.{name}"
        if full_name in stats:
            hooks.append(module.register_forward_hook(make_hook(full_name)))

    iterator = range(0, len(calibration_prompts), xtx_batch_size)
    if show_progress:
        iterator = tqdm(iterator, desc="Collect XTX", unit="batch")

    was_training = pipe.unet.training
    pipe.unet.eval()

    try:
        for start in iterator:
            batch_prompts = list(calibration_prompts[start:start + xtx_batch_size])
            batch_generator = None
            if generator_seed is not None:
                batch_generator = [
                    torch.Generator(device=gen_device).manual_seed(int(generator_seed) + start + i)
                    for i in range(len(batch_prompts))
                ]
            _ = pipe(
                batch_prompts,
                generator=batch_generator,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                height=height,
                width=width,
                output_type="latent",
            )
    finally:
        for h in hooks:
            h.remove()
        if was_training:
            pipe.unet.train()

    for full_name, item in stats.items():
        if item["n_samples"] == 0:
            raise RuntimeError(f"No calibration samples collected for layer {full_name}")
        item["xtx"] /= item["n_samples"]

    return stats


def _layer_xtx_cols(layer: nn.Module) -> int:
    if isinstance(layer, nn.Linear):
        return int(layer.in_features)
    if isinstance(layer, nn.Conv2d):
        return int(layer.in_channels * layer.kernel_size[0] * layer.kernel_size[1])
    raise TypeError(type(layer))


def _target_has_large_conv(layers: Dict[str, nn.Module]) -> bool:
    return any(isinstance(layer, nn.Conv2d) and _layer_xtx_cols(layer) >= 8192 for layer in layers.values())


def estimate_xtx_bytes_for_layers(layers: Dict[str, nn.Module], dtype: torch.dtype) -> int:
    dtype_bytes = torch.tensor([], dtype=dtype).element_size()
    return sum(_layer_xtx_cols(layer) ** 2 * dtype_bytes for layer in layers.values())


def resolve_vqdm_xtx_batch_size(
    *,
    layers: Dict[str, nn.Module],
    num_prompts: int,
    requested_batch: int = 0,
    dtype: torch.dtype = torch.float64,
    device=None,
) -> int:
    requested_batch = int(requested_batch or 0)
    has_large_conv = _target_has_large_conv(layers)
    conv_cap = max(1, VQDM_CONV_XTX_MAX_BATCH)
    max_batch_for_layers = min(num_prompts, conv_cap) if has_large_conv else num_prompts
    if requested_batch > 0 or not (VQDM_GPU_AUTOTUNE and torch.cuda.is_available()):
        batch = max(1, min(max_batch_for_layers, requested_batch if requested_batch > 0 else 1))
        if has_large_conv and requested_batch > batch:
            log_event(f"Clamped requested VQDM XTX batch {requested_batch} -> {batch} for Conv2d targets")
        return batch

    device = torch.device(device if device is not None else vqdm_primary_device)
    info = _cuda_mem_info(device)
    if info is None:
        return 1

    xtx_gib = estimate_xtx_bytes_for_layers(layers, dtype) / 1024**3
    total_gib = info["total_gib"]
    free_gib = info["free_gib"]
    # Diffusion activations dominate the prompt batch, while XTX buffers dominate large conv targets.
    per_prompt_gib = VQDM_XTX_ESTIMATED_GIB_PER_PROMPT
    budget_gib = max(1.0, total_gib * VQDM_GPU_TARGET_MEM_FRACTION - (total_gib - free_gib) - xtx_gib)
    estimated = max(1, int(budget_gib // max(per_prompt_gib, 0.1)))
    max_batch = num_prompts if VQDM_XTX_MAX_BATCH <= 0 else min(num_prompts, VQDM_XTX_MAX_BATCH)
    if has_large_conv:
        max_batch = min(max_batch, max(1, VQDM_CONV_XTX_MAX_BATCH))
    batch = max(1, min(num_prompts, max_batch, estimated))
    log_event(
        f"Auto XTX batch estimate: batch={batch}, prompts={num_prompts}, device={device}, xtx={xtx_gib:.2f} GiB, "
        f"free={free_gib:.2f} GiB, total={total_gib:.2f} GiB, large_conv={has_large_conv}"
    )
    return batch


@torch.no_grad()
def _probe_vqdm_xtx_batch(
    pipe,
    calibration_prompts,
    layer_names,
    batch: int,
    *,
    num_inference_steps: int,
    guidance_scale: float,
    height=None,
    width=None,
    collect_device="cuda",
    xtx_store_device="cuda",
    accum_dtype=torch.float64,
    group_channels=True,
    generator_seed: Optional[int] = None,
) -> bool:
    try:
        probe_prompts = list(calibration_prompts[:batch])
        stats = collect_midblock_xtx_streaming(
            pipe=pipe,
            calibration_prompts=probe_prompts,
            layer_names=layer_names,
            xtx_batch_size=batch,
            num_inference_steps=max(1, min(num_inference_steps, VQDM_XTX_PROBE_STEPS)),
            guidance_scale=guidance_scale,
            height=height,
            width=width,
            collect_device=collect_device,
            xtx_store_device=xtx_store_device,
            accum_dtype=accum_dtype,
            group_channels=group_channels,
            show_progress=False,
            generator_seed=generator_seed,
        )
        del stats
        _clear_cuda_after_oom()
        return True
    except BaseException as exc:
        if _is_cuda_oom(exc):
            _clear_cuda_after_oom()
            return False
        raise


def binary_search_vqdm_xtx_batch(
    pipe,
    calibration_prompts,
    layer_names,
    *,
    low: int,
    high: int,
    **probe_kwargs,
) -> int:
    if high <= low:
        return max(1, high)
    best = low
    lo, hi = low, high
    while lo <= hi:
        mid = (lo + hi) // 2
        ok = _probe_vqdm_xtx_batch(
            pipe,
            calibration_prompts,
            layer_names,
            mid,
            **probe_kwargs,
        )
        log_event(f"XTX batch probe batch={mid} -> {'ok' if ok else 'OOM'}")
        if ok:
            best = mid
            lo = mid + 1
        else:
            hi = mid - 1
    return max(1, best)


def _even_prompt_chunks(
    prompts: Sequence[str],
    n_chunks: int,
    weights: Optional[Sequence[float]] = None,
) -> List[Tuple[int, List[str]]]:
    n = len(prompts)
    chunks = []
    if weights is not None:
        slices = _weighted_slices(n, list(weights)[:max(1, int(n_chunks))])
    else:
        slices = [
            slice((n * idx) // max(1, int(n_chunks)), (n * (idx + 1)) // max(1, int(n_chunks)))
            for idx in range(max(1, int(n_chunks)))
        ]
    for sl in slices:
        start, end = int(sl.start or 0), int(sl.stop or 0)
        if start < end:
            chunks.append((start, list(prompts[start:end])))
    return chunks


def replicate_text2image_pipeline_for_devices(
    pipeline: StableDiffusionPipeline,
    devices: Sequence[torch.device],
    *,
    detach: bool = True,
    reuse_first_replica: bool = True,
) -> Sequence[StableDiffusionPipeline]:
    module_dict = nn.ModuleDict({k: v for k, v in pipeline.components.items() if isinstance(v, nn.Module)})
    component_replicas = nn.parallel.replicate(module_dict, devices=devices, detach=detach)
    replicas = []
    if reuse_first_replica:
        replicas.append(pipeline)
    for replica_index in range(int(reuse_first_replica), len(devices)):
        memo = {}
        for key in pipeline.components:
            original_component = pipeline.components[key]
            if key in module_dict:
                memo[id(original_component)] = component_replicas[replica_index][key]
            elif key.startswith("tokenizer"):
                memo[id(original_component)] = original_component
        replicas.append(copy.deepcopy(pipeline, memo=memo))
    return replicas


@torch.no_grad()
def collect_midblock_xtx_streaming_parallel(
    *,
    pipe,
    calibration_prompts,
    layer_names,
    xtx_batch_size,
    num_inference_steps,
    guidance_scale,
    layers: Optional[Dict[str, nn.Module]] = None,
    height=None,
    width=None,
    collect_devices: Sequence[torch.device],
    xtx_store_device="cuda",
    accum_dtype=torch.float64,
    group_channels=True,
    show_progress=True,
    generator_seed: Optional[int] = None,
    auto_per_device_batch: bool = True,
):
    collect_devices = [torch.device(d) for d in collect_devices]
    if len(collect_devices) <= 1 or len(calibration_prompts) < 2:
        return collect_midblock_xtx_streaming(
            pipe=pipe,
            calibration_prompts=calibration_prompts,
            layer_names=layer_names,
            xtx_batch_size=max(1, min(int(xtx_batch_size), len(calibration_prompts))),
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            height=height,
            width=width,
            collect_device=str(collect_devices[0]),
            xtx_store_device=xtx_store_device,
            accum_dtype=accum_dtype,
            group_channels=group_channels,
            show_progress=show_progress,
            generator_seed=generator_seed,
        )

    device_weights = [_device_vram_weight(dev) for dev in collect_devices]
    chunks = _even_prompt_chunks(calibration_prompts, len(collect_devices), weights=device_weights)
    collect_devices = collect_devices[:len(chunks)]
    log_event(f"Parallel XTX collection on {[str(d) for d in collect_devices]}")
    replicas = None
    partials = None
    try:
        replicas = replicate_text2image_pipeline_for_devices(
            force_pipeline_device_dtype(pipe, str(collect_devices[0]), runtime_dtype),
            collect_devices,
            detach=True,
            reuse_first_replica=True,
        )
        batch_cap = max(1, int(xtx_batch_size))

        def worker(replica, device, start, chunk_prompts):
            if auto_per_device_batch and layers is not None:
                local_auto = resolve_vqdm_xtx_batch_size(
                    layers=layers,
                    num_prompts=len(chunk_prompts),
                    requested_batch=0,
                    dtype=accum_dtype,
                    device=device,
                )
                local_batch = max(1, min(int(local_auto), len(chunk_prompts)))
            else:
                local_batch = max(1, min(batch_cap, len(chunk_prompts)))
            while True:
                try:
                    log_event(f"Collect XTX shard on {device}: prompts={len(chunk_prompts)}, batch={local_batch}")
                    return collect_midblock_xtx_streaming(
                        pipe=replica,
                        calibration_prompts=chunk_prompts,
                        layer_names=layer_names,
                        xtx_batch_size=local_batch,
                        num_inference_steps=num_inference_steps,
                        guidance_scale=guidance_scale,
                        height=height,
                        width=width,
                        collect_device=str(device),
                        xtx_store_device=str(device),
                        accum_dtype=accum_dtype,
                        group_channels=group_channels,
                        show_progress=False,
                        generator_seed=None if generator_seed is None else int(generator_seed) + start,
                    )
                except BaseException as exc:
                    if _is_cuda_oom(exc) and local_batch > 1:
                        old_batch = local_batch
                        local_batch = max(1, local_batch // 2)
                        _clear_cuda_after_oom()
                        log_event(f"CUDA OOM on XTX shard {device}; reducing batch {old_batch} -> {local_batch}")
                        continue
                    raise

        with ThreadPoolExecutor(max_workers=len(chunks)) as executor:
            futures = [
                executor.submit(worker, replicas[i], collect_devices[i], start, chunk_prompts)
                for i, (start, chunk_prompts) in enumerate(chunks)
            ]
            partials = [future.result() for future in futures]

        merged = {}
        target_device = torch.device(xtx_store_device)
        for full_name in layer_names:
            total_n = sum(int(part[full_name]["n_samples"]) for part in partials)
            if total_n <= 0:
                raise RuntimeError(f"No calibration samples collected for layer {full_name}")
            acc = None
            for part in partials:
                n = int(part[full_name]["n_samples"])
                xtx = part[full_name]["xtx"].to(device=target_device, dtype=accum_dtype)
                weighted = xtx * n
                acc = weighted if acc is None else acc + weighted
            merged[full_name] = {
                "xtx": acc / total_n,
                "n_samples": total_n,
                "layer": partials[0][full_name]["layer"],
            }
        return merged
    finally:
        del replicas, partials
        gc.collect()
        _clear_cuda_after_oom()


@torch.no_grad()
def collect_midblock_xtx_streaming_autobatch(
    *,
    pipe,
    calibration_prompts,
    layers: Dict[str, nn.Module],
    layer_names,
    xtx_batch_size=0,
    num_inference_steps=50,
    guidance_scale=5.0,
    height=None,
    width=None,
    collect_devices=None,
    xtx_store_device="cuda",
    accum_dtype=torch.float64,
    group_channels=True,
    show_progress=True,
    generator_seed: Optional[int] = None,
):
    collect_devices = collect_devices or [torch.device(vqdm_collect_device)]
    requested = int(xtx_batch_size or 0)
    start_batch = resolve_vqdm_xtx_batch_size(
        layers=layers,
        num_prompts=len(calibration_prompts),
        requested_batch=requested,
        dtype=accum_dtype,
        device=collect_devices[0],
    )

    if requested <= 0 and VQDM_XTX_BINSEARCH and torch.cuda.is_available():
        start_batch = binary_search_vqdm_xtx_batch(
            pipe,
            calibration_prompts,
            layer_names,
            low=1,
            high=start_batch,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            height=height,
            width=width,
            collect_device=str(collect_devices[0]),
            xtx_store_device=xtx_store_device,
            accum_dtype=accum_dtype,
            group_channels=group_channels,
            generator_seed=generator_seed,
        )

    batch = max(1, int(start_batch))
    while True:
        try:
            log_event(f"Collect XTX with batch={batch}, devices={[str(d) for d in collect_devices]}")
            return collect_midblock_xtx_streaming_parallel(
                pipe=pipe,
                calibration_prompts=calibration_prompts,
                layers=layers,
                layer_names=layer_names,
                xtx_batch_size=batch,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                height=height,
                width=width,
                collect_devices=collect_devices,
                xtx_store_device=xtx_store_device,
                accum_dtype=accum_dtype,
                group_channels=group_channels,
                show_progress=show_progress,
                generator_seed=generator_seed,
                auto_per_device_batch=(requested <= 0 and VQDM_GPU_AUTOTUNE),
            )
        except BaseException as exc:
            if _is_cuda_oom(exc) and batch > 1:
                old_batch = batch
                batch = max(1, batch // 2)
                _clear_cuda_after_oom()
                log_event(f"CUDA OOM during XTX collection; reducing batch {old_batch} -> {batch}")
                continue
            raise

# %% [markdown] cell 32
# <h1>Запуск квантизации</h1>

# %% cell 33
@torch.no_grad()
def quantize_layer_aq_from_engine(
    engine: AQEngine,
    cfg: AQConfig,
    group_channels: bool = True,
    verbose: bool = True,
) -> nn.Module:
    original_layer = engine.layer
    qweight = engine.quantize(cfg=cfg, verbose=verbose)

    if isinstance(original_layer, nn.Linear):
        qlayer = QuantizedLinear(
            quantized_weight=qweight,
            bias=original_layer.bias,
        )
    elif isinstance(original_layer, nn.Conv2d):
        qlayer = QuantizedConv2D.from_conv(
            quantized_weight=qweight,
            conv=original_layer,
            group_channels=group_channels,
        )
    else:
        raise TypeError(type(original_layer))

    qlayer = qlayer.to(device=original_layer.weight.device, dtype=original_layer.weight.dtype)
    return qlayer

# %% cell 34
@torch.no_grad()
def quantize_mid_block_vqdm(
    pipe_fp: StableDiffusionPipeline,
    calibration_prompts: Sequence[str],
    cfg: AQConfig,
    layer_names: Optional[Sequence[str]] = None,
    xtx_batch_size: int = 4,
    num_inference_steps: int = 50,
    guidance_scale: float = 5.0,
    height: Optional[int] = None,
    width: Optional[int] = None,
    min_features: int = 16,
    group_channels: bool = True,
    include_linear: bool = True,
    include_conv: bool = True,
    collect_device: str = "cuda",
    xtx_store_device: str = "cpu",
    xtx_stats: Optional[Dict[str, Dict[str, Any]]] = None,
    xtx_generator_seed: Optional[int] = None,
    final_device: str = "cpu",
    final_dtype: torch.dtype = torch.float32,
    verbose: bool = True,
) -> StableDiffusionPipeline:
    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)

    quant_device = cfg.quant_device if cfg.quant_device != "auto" else collect_device

    initial_device = collect_device if xtx_stats is None else quant_device
    source_device = next(pipe_fp.unet.parameters()).device
    if vqdm_cpu_copy_before_pipe_clone and str(source_device).startswith("cuda"):
        if verbose:
            print("[VQDM] Moving source pipeline to CPU before deepcopy to avoid a temporary double-GPU copy")
        move_pipeline_to_cpu_safely(pipe_fp)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    initial_dtype = globals().get("runtime_dtype", torch.float16) if str(initial_device).startswith("cuda") else torch.float32
    pipe_q = force_pipeline_device_dtype(copy.deepcopy(pipe_fp), initial_device, initial_dtype)

    all_layers = get_mid_block_quant_layers(
        pipe_q.unet,
        min_features=min_features,
        include_linear=include_linear,
        include_conv=include_conv,
    )

    quant_order = list(all_layers.keys()) if layer_names is None else list(layer_names)

    if xtx_stats is None:
        if verbose:
            print(f"[VQDM] Collect XTX for {len(quant_order)} layers | collect_device={collect_device} | xtx_store={xtx_store_device} | seed={xtx_generator_seed}")

        stats = collect_midblock_xtx_streaming_autobatch(
            pipe=pipe_q,
            calibration_prompts=calibration_prompts,
            layers={name: all_layers[name] for name in quant_order},
            layer_names=quant_order,
            xtx_batch_size=xtx_batch_size,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            height=height,
            width=width,
            collect_devices=globals().get("vqdm_collect_devices", [torch.device(collect_device)]),
            xtx_store_device=xtx_store_device,
            accum_dtype=cfg.accum_dtype,
            group_channels=group_channels,
            show_progress=verbose,
            generator_seed=xtx_generator_seed,
        )
    else:
        stats = xtx_stats
        missing = [name for name in quant_order if name not in stats]
        if missing:
            raise KeyError(f"Precomputed XTX is missing layers: {missing}")
        if verbose:
            print(f"[VQDM] Reusing precomputed XTX for {len(quant_order)} layers")

    quant_dtype = globals().get("runtime_dtype", torch.float16) if str(quant_device).startswith("cuda") else torch.float32
    pipe_q = force_pipeline_device_dtype(pipe_q, quant_device, quant_dtype)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if verbose:
        print(f"[VQDM] Start quantization of {len(quant_order)} layers | quant_device={quant_device}")

    iterator = quant_order
    if verbose:
        iterator = tqdm(iterator, desc="Sequential AQ", unit="layer")

    for full_name in iterator:
        assert full_name.startswith("mid_block.")
        local_name = full_name[len("mid_block."):]

        current_layer = get_module_by_name(pipe_q.unet.mid_block, local_name)

        if isinstance(current_layer, (QuantizedLinear, QuantizedConv2D)):
            if verbose:
                print(f"[VQDM] Skipping already quantized layer: {full_name}")
            continue

        if full_name not in stats:
            raise KeyError(f"No collected XTX for layer {full_name}")

        current_layer = current_layer.to(quant_device)

        engine = AQEngine(
            current_layer,
            accum_dtype=cfg.accum_dtype,
            group_channels=group_channels,
            xtx_device=quant_device,
        )

        engine.XTX.copy_(stats[full_name]["xtx"].to(device=engine.XTX.device, dtype=engine.XTX.dtype))
        engine.nsamples = stats[full_name]["n_samples"]

        if verbose:
            print(
                f"[VQDM] Quantizing {full_name} "
                f"| weight={tuple(current_layer.weight.shape)} "
                f"| nsamples={engine.nsamples} "
                f"| xtx_shape={tuple(engine.XTX.shape)} "
                f"| xtx_device={engine.XTX.device}"
            )

        t0 = time.perf_counter()
        qlayer = quantize_layer_aq_from_engine(
            engine,
            cfg=cfg,
            group_channels=group_channels,
            verbose=verbose,
        )
        elapsed = time.perf_counter() - t0
        if verbose and hasattr(iterator, "set_postfix"):
            iterator.set_postfix_str(f"{full_name} | {elapsed:.1f}s")

        set_module_by_name(pipe_q.unet.mid_block, local_name, qlayer)

        del engine
        del qlayer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    pipe_q = force_pipeline_device_dtype(pipe_q, final_device, final_dtype)
    return pipe_q


class _MidBlockHolder(nn.Module):
    def __init__(self, mid_block: nn.Module):
        super().__init__()
        self.mid_block = mid_block


@torch.no_grad()
def quantize_mid_block_module_vqdm(
    pipe_fp: StableDiffusionPipeline,
    calibration_prompts: Sequence[str],
    cfg: AQConfig,
    layer_names: Optional[Sequence[str]] = None,
    xtx_batch_size: int = 4,
    num_inference_steps: int = 50,
    guidance_scale: float = 5.0,
    height: Optional[int] = None,
    width: Optional[int] = None,
    min_features: int = 16,
    group_channels: bool = True,
    include_linear: bool = True,
    include_conv: bool = True,
    collect_device: str = "cuda",
    xtx_store_device: str = "cpu",
    xtx_stats: Optional[Dict[str, Dict[str, Any]]] = None,
    xtx_generator_seed: Optional[int] = None,
    final_device: str = "cpu",
    final_dtype: torch.dtype = torch.float32,
    verbose: bool = True,
) -> nn.Module:
    """Quantize only UNet.mid_block, matching the RVQ runner's low-overhead eval path."""
    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)

    quant_device = cfg.quant_device if cfg.quant_device != "auto" else collect_device
    initial_device = collect_device if xtx_stats is None else quant_device
    initial_dtype = globals().get("runtime_dtype", torch.float16) if str(initial_device).startswith("cuda") else torch.float32

    src_mid_block = pipe_fp.unet.mid_block
    src_param = next(src_mid_block.parameters(), None)
    source_device = src_param.device if src_param is not None else torch.device("cpu")
    source_dtype = src_param.dtype if src_param is not None else torch.float32

    if vqdm_cpu_deepcopy_before_quant and source_device.type != "cpu":
        if verbose:
            print(f"[VQDM] Moving source mid_block from {source_device} to CPU before deepcopy (low-VRAM path)")
        src_mid_block.to("cpu", dtype=torch.float32)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        mid_block_q = copy.deepcopy(src_mid_block).to(initial_device, dtype=initial_dtype)
        src_mid_block.to(source_device, dtype=source_dtype)
    else:
        if str(source_device) != str(initial_device):
            log_event(f"[VQDM] Keeping source mid_block on {source_device} for deepcopy; work_device={initial_device}")
        mid_block_q = copy.deepcopy(src_mid_block).to(initial_device, dtype=initial_dtype)

    holder = _MidBlockHolder(mid_block_q)
    all_layers = get_mid_block_quant_layers(
        holder,
        min_features=min_features,
        include_linear=include_linear,
        include_conv=include_conv,
    )
    quant_order = list(all_layers.keys()) if layer_names is None else list(layer_names)
    missing_from_default_filter = [name for name in quant_order if name not in all_layers]
    if missing_from_default_filter:
        all_midblock_layers = get_all_mid_block_quant_layers(
            holder,
            min_features=min_features,
            include_linear=include_linear,
            include_conv=True,
        )
        recoverable = {name: all_midblock_layers[name] for name in missing_from_default_filter if name in all_midblock_layers}
        all_layers.update(recoverable)
        still_missing = [name for name in missing_from_default_filter if name not in all_layers]
        if still_missing:
            raise KeyError(f"Requested VQDM layer(s) were not found in mid_block: {still_missing}")

    if xtx_stats is None:
        if verbose:
            print(f"[VQDM] Collect XTX for {len(quant_order)} layers | collect_device={collect_device} | xtx_store={xtx_store_device} | seed={xtx_generator_seed}")

        stats = collect_midblock_xtx_streaming_autobatch(
            pipe=pipe_fp,
            calibration_prompts=calibration_prompts,
            layers={name: all_layers[name] for name in quant_order},
            layer_names=quant_order,
            xtx_batch_size=xtx_batch_size,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            height=height,
            width=width,
            collect_devices=globals().get("vqdm_collect_devices", [torch.device(collect_device)]),
            xtx_store_device=xtx_store_device,
            accum_dtype=cfg.accum_dtype,
            group_channels=group_channels,
            show_progress=verbose,
            generator_seed=xtx_generator_seed,
        )
    else:
        stats = xtx_stats
        missing = [name for name in quant_order if name not in stats]
        if missing:
            raise KeyError(f"Precomputed XTX is missing layers: {missing}")
        if verbose:
            print(f"[VQDM] Reusing precomputed XTX for {len(quant_order)} layers")

    quant_dtype = globals().get("runtime_dtype", torch.float16) if str(quant_device).startswith("cuda") else torch.float32
    mid_block_q = mid_block_q.to(quant_device, dtype=quant_dtype)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if verbose:
        print(f"[VQDM] Start quantization of {len(quant_order)} layers | quant_device={quant_device}")

    iterator = quant_order
    if verbose:
        iterator = tqdm(iterator, desc="Sequential AQ", unit="layer")

    for full_name in iterator:
        assert full_name.startswith("mid_block.")
        local_name = full_name[len("mid_block."):]
        current_layer = get_module_by_name(mid_block_q, local_name)

        if isinstance(current_layer, (QuantizedLinear, QuantizedConv2D)):
            if verbose:
                print(f"[VQDM] Skipping already quantized layer: {full_name}")
            continue

        if full_name not in stats:
            raise KeyError(f"No collected XTX for layer {full_name}")

        current_layer = current_layer.to(quant_device)
        engine = AQEngine(
            current_layer,
            accum_dtype=cfg.accum_dtype,
            group_channels=group_channels,
            xtx_device=quant_device,
        )
        engine.XTX.copy_(stats[full_name]["xtx"].to(device=engine.XTX.device, dtype=engine.XTX.dtype))
        engine.nsamples = stats[full_name]["n_samples"]

        if verbose:
            print(
                f"[VQDM] Quantizing {full_name} "
                f"| weight={tuple(current_layer.weight.shape)} "
                f"| nsamples={engine.nsamples} "
                f"| xtx_shape={tuple(engine.XTX.shape)} "
                f"| xtx_device={engine.XTX.device}"
            )

        t0 = time.perf_counter()
        qlayer = quantize_layer_aq_from_engine(
            engine,
            cfg=cfg,
            group_channels=group_channels,
            verbose=verbose,
        )
        elapsed = time.perf_counter() - t0
        if verbose and hasattr(iterator, "set_postfix"):
            iterator.set_postfix_str(f"{full_name} | {elapsed:.1f}s")

        set_module_by_name(mid_block_q, local_name, qlayer)
        del engine, qlayer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    mid_block_q = mid_block_q.to(final_device, dtype=final_dtype)
    return mid_block_q


def build_vqdm_midblock_pipe_from_cfg(
    profile: Dict[str, Any],
    cfg: AQConfig,
    label: str,
    *,
    target_layer_names: Optional[Sequence[str]] = None,
    include_conv: Optional[bool] = None,
    xtx_stats: Optional[Dict[str, Dict[str, Any]]] = None,
):
    layer_names = list(target_layer_names) if target_layer_names is not None else list(vqdm_target_layer_names)
    include_conv_for_scope = bool(vqdm_include_conv if include_conv is None else include_conv)
    stats_for_scope = vqdm_xtx_stats if xtx_stats is None else xtx_stats

    log_section(f"Quantizing {label}")
    log_event(
        f"VQDM config: M={cfg.num_codebooks}, bits={cfg.nbits_per_codebook}, "
        f"K={2 ** cfg.nbits_per_codebook}, in_group={cfg.in_group_size}, "
        f"beam={cfg.beam_size}, epochs={cfg.max_epochs}, steps_per_epoch={cfg.steps_per_epoch}, "
        f"eval_inplace_midblock={vqdm_eval_inplace_midblock}, layers={len(layer_names)}, "
        f"include_conv={include_conv_for_scope}"
    )
    log_cuda_memory("before VQDM quantization", vqdm_primary_device)
    t0 = time.time()
    mid_block_q = quantize_mid_block_module_vqdm(
        pipe_fp=pipe,
        calibration_prompts=vqdm_calibration_prompts,
        cfg=cfg,
        layer_names=layer_names,
        xtx_batch_size=vqdm_xtx_batch_size,
        num_inference_steps=vqdm_xtx_num_inference_steps,
        guidance_scale=guidance_scale,
        min_features=16,
        group_channels=True,
        include_linear=True,
        include_conv=include_conv_for_scope,
        collect_device=vqdm_collect_device,
        xtx_store_device=vqdm_xtx_store_device,
        xtx_stats=stats_for_scope,
        xtx_generator_seed=vqdm_xtx_seed,
        final_device=vqdm_final_device,
        final_dtype=vqdm_final_dtype,
        verbose=True,
    )
    print(f"{label} quantized in {time.time() - t0:.1f}s")
    log_cuda_memory("after VQDM quantization", vqdm_primary_device)

    if vqdm_eval_inplace_midblock:
        log_event("Skipping full pipeline clone; quantized VQDM mid_block will be evaluated in-place")
        return None, mid_block_q

    pipe_base_device = vqdm_eval_device if vqdm_keep_eval_pipes_on_device else "cpu"
    pipe_base_dtype = vqdm_final_dtype if pipe_base_device == vqdm_eval_device else torch.float32

    if str(pipe_base_device).startswith("cuda"):
        if vqdm_cpu_copy_before_pipe_clone:
            move_pipeline_to_cpu_safely(pipe)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            pipe_q = force_pipeline_device_dtype(copy.deepcopy(pipe), pipe_base_device, pipe_base_dtype)
            if vqdm_keep_eval_pipes_on_device:
                force_pipeline_device_dtype(pipe, pipe_base_device, pipe_base_dtype)
        else:
            if _module_device(pipe.unet).type != "cuda":
                force_pipeline_device_dtype(pipe, pipe_base_device, pipe_base_dtype)
            log_event("Cloning FP pipeline directly on CUDA; CPU pre-copy disabled")
            pipe_q = force_pipeline_device_dtype(copy.deepcopy(pipe), pipe_base_device, pipe_base_dtype)
    else:
        pipe_fp_local = move_pipeline_to_cpu_safely(pipe)
        pipe_q = copy.deepcopy(pipe_fp_local)

    pipe_q.unet.mid_block = mid_block_q.to(pipe_base_device, dtype=pipe_base_dtype)
    optimize_pipeline_for_inference(pipe_q, label=label)
    if vqdm_keep_eval_pipes_on_device:
        optimize_pipeline_for_inference(pipe, label="FP pipeline")

    if not vqdm_keep_eval_pipes_on_device:
        move_pipeline_to_cpu_safely(pipe)
        move_pipeline_to_cpu_safely(pipe_q)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    log_cuda_memory("after VQDM quantized pipeline build", vqdm_primary_device)
    return pipe_q, mid_block_q

# %% [markdown] cell 35
# <h1>Check</h1>

# %% cell 36
# PickScore train/val CSVs are the only prompt source for comparable runs.
# See load_pickscore_reference_prompts(...) below.


# %% cell 37
def count_quantized_layers_in_midblock(unet: nn.Module):
    total_linear = 0
    total_conv = 0
    total_qlinear = 0
    total_qconv = 0
    quant_names = []

    for name, module in unet.mid_block.named_modules():
        full_name = f"mid_block.{name}" if name else "mid_block"

        if isinstance(module, nn.Linear):
            total_linear += 1
        elif isinstance(module, nn.Conv2d):
            total_conv += 1
        elif isinstance(module, QuantizedLinear):
            total_qlinear += 1
            quant_names.append(full_name)
        elif isinstance(module, QuantizedConv2D):
            total_qconv += 1
            quant_names.append(full_name)

    return {
        "total_linear_left": total_linear,
        "total_conv_left": total_conv,
        "total_quantized_linear": total_qlinear,
        "total_quantized_conv": total_qconv,
        "quantized_names": quant_names,
    }

# %% cell 38
@torch.no_grad()
def generate_one_image(
    pipe: StableDiffusionPipeline,
    prompt: str,
    seed: int = 1234,
    num_inference_steps: int = 50,
    guidance_scale: float = 5.0,
    height: Optional[int] = None,
    width: Optional[int] = None,
    device: str = None,
    dtype: torch.dtype = None,
):
    device = device or vqdm_eval_device
    dtype = dtype or (runtime_dtype if str(device).startswith("cuda") else torch.float32)
    pipe = force_pipeline_device_dtype(pipe, device, dtype)
    pipe.set_progress_bar_config(disable=True)
    gen_device = device if str(device).startswith("cuda") else "cpu"
    gen = torch.Generator(device=gen_device).manual_seed(seed)

    out = pipe(
        prompt,
        generator=gen,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        height=height,
        width=width,
    )

    img = out.images[0]

    if not globals().get("vqdm_keep_eval_pipes_on_device", True):
        move_pipeline_to_cpu_safely(pipe)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return img


@torch.no_grad()
def sanity_check_vqdm_pipe(
    pipe_fp: StableDiffusionPipeline,
    pipe_q: StableDiffusionPipeline,
    prompt: str,
    seed: int = 1234,
    num_inference_steps: int = 50,
    guidance_scale: float = 5.0,
    height: Optional[int] = None,
    width: Optional[int] = None,
    device: str = None,
    show_images: bool = True,
):
    device = device or vqdm_eval_device
    print("=" * 100)
    print("SANITY CHECK: structure")
    stats = count_quantized_layers_in_midblock(pipe_q.unet)
    print(f"QuantizedLinear in mid_block: {stats['total_quantized_linear']}")
    print(f"QuantizedConv2D in mid_block: {stats['total_quantized_conv']}")
    print(f"Plain nn.Linear left in mid_block: {stats['total_linear_left']}")
    print(f"Plain nn.Conv2d left in mid_block: {stats['total_conv_left']}")

    if stats["total_quantized_linear"] + stats["total_quantized_conv"] == 0:
        raise RuntimeError("No quantized layers found in mid_block")

    print("\nFirst quantized layers:")
    for name in stats["quantized_names"][:10]:
        print(" ", name)

    print("\n" + "=" * 100)
    print("SANITY CHECK: generation")

    t0 = time.perf_counter()
    img_fp = generate_one_image(
        pipe_fp,
        prompt,
        seed=seed,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        height=height,
        width=width,
        device=device,
    )
    t_fp = time.perf_counter() - t0
    print(f"FP generation ok   | size={img_fp.size} | time={t_fp:.2f}s")

    t0 = time.perf_counter()
    img_q = generate_one_image(
        pipe_q,
        prompt,
        seed=seed,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        height=height,
        width=width,
        device=device,
    )
    t_q = time.perf_counter() - t0
    print(f"Q generation ok    | size={img_q.size} | time={t_q:.2f}s")

    if show_images:
        fig = plt.figure(figsize=(10, 5))

        plt.subplot(1, 2, 1)
        plt.imshow(img_fp)
        plt.title("FP")
        plt.axis("off")

        plt.subplot(1, 2, 2)
        plt.imshow(img_q)
        plt.title("VQDM-like AQ")
        plt.axis("off")

        plt.tight_layout()
        plt.show()

    return {
        "img_fp": img_fp,
        "img_q": img_q,
        "stats": stats,
        "t_fp": t_fp,
        "t_q": t_q,
    }

# %% cell 39
# === VQDM bpp sweep setup: target layer and one shared XTX ===

# Effective settings from the official VQDM text2image quantization example,
# adapted to SD1.5 and to the currently selected subset of layers.
num_inference_steps = 50
guidance_scale = 5.0
seed = 1234

VQDM_HARDCODED_MIDBLOCK_LINEAR_NAMES = [
    "mid_block.attentions.0.transformer_blocks.0.attn1.to_q",
    "mid_block.attentions.0.transformer_blocks.0.attn1.to_k",
    "mid_block.attentions.0.transformer_blocks.0.attn1.to_v",
    "mid_block.attentions.0.transformer_blocks.0.attn1.to_out.0",
    "mid_block.attentions.0.transformer_blocks.0.attn2.to_q",
    "mid_block.attentions.0.transformer_blocks.0.attn2.to_k",
    "mid_block.attentions.0.transformer_blocks.0.attn2.to_v",
    "mid_block.attentions.0.transformer_blocks.0.attn2.to_out.0",
    "mid_block.attentions.0.transformer_blocks.0.ff.net.0.proj",
    "mid_block.attentions.0.transformer_blocks.0.ff.net.2",
]
vqdm_single_target_layer_name = (
    VQDM_SINGLE_TARGET_LAYER
    or "mid_block.attentions.0.transformer_blocks.0.ff.net.0.proj"
)

# VQDM/AQLM bpp schedule: keep 8-bit codebooks and change the number of additive codebooks M.
# With in_group_size=8, index bpp is M * 8 / 8 = M; codebooks/scales add the ~0.15-ish overhead.
vqdm_bpp_profiles = [
    {"key": "4.15", "label": "VQDM target~4.15bpp", "reference_bpp": 4.15, "num_codebooks": 4, "nbits_per_codebook": 8},
    {"key": "3.15", "label": "VQDM target~3.15bpp", "reference_bpp": 3.15, "num_codebooks": 3, "nbits_per_codebook": 8},
    {"key": "2.15", "label": "VQDM target~2.15bpp", "reference_bpp": 2.15, "num_codebooks": 2, "nbits_per_codebook": 8},
]

expected_vqdm_bpp_profiles = {
    "4.15": {"num_codebooks": 4, "nbits_per_codebook": 8},
    "3.15": {"num_codebooks": 3, "nbits_per_codebook": 8},
    "2.15": {"num_codebooks": 2, "nbits_per_codebook": 8},
}
for profile in vqdm_bpp_profiles:
    expected = expected_vqdm_bpp_profiles.get(profile["key"])
    if expected is None:
        raise ValueError(f"Unexpected VQDM bpp profile key: {profile['key']}")
    for field, value in expected.items():
        if int(profile[field]) != int(value):
            raise ValueError(
                f"VQDM profile {profile['key']} must use {field}={value}; got {profile[field]}"
            )


def _pickscore_dataset_roots(*, env_name: str) -> List[Path]:
    roots: List[Path] = []
    for value in (os.environ.get(env_name), os.environ.get("PICKSCORE_DATA_DIR")):
        if value:
            roots.append(Path(value).expanduser())

    script_dir = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
    roots.extend(
        [
            Path.cwd() / "vqdm_ref_repo" / "dataset",
            script_dir / "vqdm_ref_repo" / "dataset",
            Path.cwd() / "dataset",
            script_dir / "dataset",
        ]
    )

    unique_roots: List[Path] = []
    seen: set[str] = set()
    for root in roots:
        key = str(root.resolve()) if root.exists() else str(root)
        if key not in seen:
            seen.add(key)
            unique_roots.append(root)
    return unique_roots


def _resolve_pickscore_csv(filename: str, *, env_name: str) -> Path:
    checked: List[Path] = []
    for root in _pickscore_dataset_roots(env_name=env_name):
        csv_path = root / filename
        checked.append(csv_path)
        if csv_path.exists():
            return csv_path

    checked_text = "\n  - ".join(str(path) for path in checked)
    raise FileNotFoundError(
        f"Required PickScore CSV was not found: {filename}\n"
        f"Set {env_name} or PICKSCORE_DATA_DIR to the directory with PickScore CSV files, "
        "or place vqdm_ref_repo/dataset next to the script.\n"
        f"Checked:\n  - {checked_text}"
    )


def load_pickscore_reference_prompts(nsamples: int, *, seed: int = 0, eval_mode: bool = False) -> List[str]:
    """Match vqdm_ref_repo.src.datautils.get_diffusion_prompts('pickscore', ...)."""
    filename = "pickscore_filtered_prompts_val.csv" if eval_mode else "pickscore_filtered_prompts_train.csv"
    csv_path = _resolve_pickscore_csv(filename, env_name="VQDM_PICKSCORE_DATA_DIR")

    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or "caption" not in reader.fieldnames:
            raise ValueError(f"PickScore CSV must contain a 'caption' column: {csv_path}")
        data = [row["caption"].strip() for row in reader if row.get("caption") and row["caption"].strip()]

    if not data:
        raise ValueError(f"PickScore CSV contains no non-empty captions: {csv_path}")
    if nsamples > len(data):
        raise ValueError(
            f"Requested {nsamples} PickScore prompts from {csv_path}, but only {len(data)} captions are available."
        )

    random.seed(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    chosen = np.random.choice(data, size=nsamples, replace=False)
    print(f"Using PickScore CSV: {csv_path} ({nsamples}/{len(data)} prompts, seed={seed}, eval_mode={eval_mode})")
    return [str(x) for x in chosen]


vqdm_use_reference_pickscore_prompts = True
vqdm_calibration_prompt_count = 256
vqdm_eval_prompt_count = 32
vqdm_eval_seeds = [1234, 4321, 999]
vqdm_xtx_full_steps = num_inference_steps
vqdm_xtx_seed = 0
vqdm_xtx_batch_size = VQDM_XTX_BATCH_ENV
vqdm_eval_num_inference_steps = num_inference_steps
vqdm_eval_guidance_scale = guidance_scale
vqdm_compute_hps = bool(VQDM_COMPUTE_HPS_REQUESTED and globals().get("HPSV2_BPE_READY", False))
vqdm_run_sanity_check = False

if vqdm_fast_smoke_test:
    num_inference_steps = 6
    vqdm_eval_num_inference_steps = 6
    vqdm_calibration_prompts = load_pickscore_reference_prompts(1, seed=0, eval_mode=False)
    vqdm_eval_prompts = load_pickscore_reference_prompts(1, seed=0, eval_mode=True)
    vqdm_eval_seeds_active = [1234]
    vqdm_xtx_num_inference_steps = 6
    vqdm_xtx_batch_size = 1
    vqdm_active_profiles = [dict(vqdm_bpp_profiles[-1], key="smoke", label="VQDM smoke~2bpp")]
    print("VQDM fast smoke mode is ON: quick functionality check, not comparable quality numbers.")
else:
    if not vqdm_use_reference_pickscore_prompts:
        raise RuntimeError("VQDM prompt selection now requires PickScore CSVs; keep vqdm_use_reference_pickscore_prompts=True.")
    vqdm_calibration_prompts = load_pickscore_reference_prompts(
        vqdm_calibration_prompt_count,
        seed=0,
        eval_mode=False,
    )
    vqdm_eval_prompts = load_pickscore_reference_prompts(
        vqdm_eval_prompt_count,
        seed=0,
        eval_mode=True,
    )
    print("Using required PickScore train/val prompts like the official VQDM script.")

    vqdm_eval_seeds_active = list(vqdm_eval_seeds)
    vqdm_xtx_num_inference_steps = vqdm_xtx_full_steps
    vqdm_active_profiles = [dict(p) for p in vqdm_bpp_profiles]
    print("VQDM comparable bpp sweep is ON: 8-bit codebooks with M=4/3/2, DDIM, guidance=5.0, 50 steps.")


def vqdm_profile_key(profile: Dict[str, Any]) -> str:
    return str(profile["key"])


def vqdm_profile_label(profile: Dict[str, Any]) -> str:
    return str(profile.get("label", vqdm_profile_key(profile)))


def make_vqdm_aq_cfg(profile: Dict[str, Any]) -> AQConfig:
    return AQConfig(
        in_group_size=8,
        out_group_size=1,
        num_codebooks=int(profile["num_codebooks"]),
        nbits_per_codebook=int(profile["nbits_per_codebook"]),
        code_dtype=torch.int32,

        groupwise_codebooks=False,
        input_channel_scales=False,
        codebook_value_nbits=16,
        codebook_value_num_groups=1,
        scale_nbits=0,

        init_max_iter=4 if vqdm_fast_smoke_test else 100,
        init_max_points_per_centroid=None,
        use_faiss=False,

        lr=1e-4,
        beam_size=1 if vqdm_fast_smoke_test else 8,
        max_epochs=1 if vqdm_fast_smoke_test else 1000,
        steps_per_epoch=1 if vqdm_fast_smoke_test else 100,
        relative_mse_tolerance=1e-2,

        seed=0,
        accum_dtype=torch.float64,
        quant_device=vqdm_quant_device,
        devices=vqdm_quant_devices,
    )


def estimate_vqdm_compression_stats(layers: Dict[str, nn.Module], profile: Dict[str, Any], cfg: AQConfig) -> Dict[str, Any]:
    weight_params = sum(layer.weight.numel() for layer in layers.values())
    bias_params = sum(layer.bias.numel() for layer in layers.values() if getattr(layer, "bias", None) is not None)
    index_bits = 0
    codebook_bits = 0
    scale_bits = 0
    input_scale_bits = 0

    for layer in layers.values():
        if isinstance(layer, nn.Linear):
            out_features, in_features = layer.weight.shape
        elif isinstance(layer, nn.Conv2d):
            out_features = layer.out_channels
            in_features = layer.in_channels * layer.kernel_size[0] * layer.kernel_size[1]
        else:
            continue

        out_groups = out_features // cfg.out_group_size
        in_groups = in_features // cfg.in_group_size
        index_bits += out_groups * in_groups * cfg.num_codebooks * cfg.nbits_per_codebook
        codebook_groups = in_groups if cfg.groupwise_codebooks else 1
        codebook_bits += (
            codebook_groups
            * cfg.num_codebooks
            * (2 ** cfg.nbits_per_codebook)
            * cfg.out_group_size
            * cfg.in_group_size
            * cfg.codebook_value_nbits
        )
        if cfg.scale_nbits >= 16 or 2 ** cfg.scale_nbits >= in_groups:
            scale_bits += cfg.scale_nbits * out_groups * in_groups
        elif 0 < cfg.scale_nbits < 16:
            scale_bits += cfg.scale_nbits * out_groups * in_groups
            scale_bits += out_groups * (2 ** cfg.scale_nbits) * 16
        elif cfg.scale_nbits == 0:
            scale_bits += out_groups * 16
        else:
            raise RuntimeError("Unexpected scale_nbits")

        if cfg.input_channel_scales:
            input_scale_bits += in_features * 32

    total_bits = index_bits + codebook_bits + scale_bits + input_scale_bits
    fp_bits = weight_params * 16
    bpp = total_bits / weight_params if weight_params else float("nan")
    compression = fp_bits / total_bits if total_bits else float("nan")

    return {
        "target_layers": len(layers),
        "target_weight_params": int(weight_params),
        "bias_params": int(bias_params),
        "num_codebooks": int(cfg.num_codebooks),
        "nbits_per_codebook": int(cfg.nbits_per_codebook),
        "codebook_size": int(2 ** cfg.nbits_per_codebook),
        "index_bits": int(index_bits),
        "codebook_bits": int(codebook_bits),
        "scale_bits": int(scale_bits),
        "input_scale_bits": int(input_scale_bits),
        "total_bits": int(total_bits),
        "target_weight_fp16_bits": int(fp_bits),
        "estimated_bpp": float(bpp),
        "estimated_compression": float(compression),
        "reference_bpp": float(profile.get("reference_bpp", float("nan"))),
    }


def print_vqdm_compression_stats(layers: Dict[str, nn.Module], profile: Dict[str, Any], cfg: AQConfig) -> Dict[str, Any]:
    stats = estimate_vqdm_compression_stats(layers, profile, cfg)

    print(f"=== {vqdm_profile_label(profile)} compression stats (selected VQDM layer set) ===")
    print(f"Target layers                    : {stats['target_layers']}")
    print(f"Target weight params             : {stats['target_weight_params']:,}")
    print(f"Bias params in target layers     : {stats['bias_params']:,}")
    print(f"Codebooks                        : {stats['num_codebooks']}")
    print(f"nbits_per_codebook               : {stats['nbits_per_codebook']}")
    print(f"Packed index bits                : {stats['index_bits']:,}")
    print(f"Codebook bits                    : {stats['codebook_bits']:,}")
    print(f"AQ scale bits                    : {stats['scale_bits']:,}")
    print(f"Input channel scale bits         : {stats['input_scale_bits']:,}")
    print(f"Target weight FP16 bits          : {stats['target_weight_fp16_bits']:,}")
    print(f"Estimated target weight bpp      : {stats['estimated_bpp']:.6f}")
    print(f"Estimated target compression     : {stats['estimated_compression']:.4f}x")
    print(f"Reference paper bpp label        : ~{stats['reference_bpp']:.2f}")
    print()
    return stats


vqdm_all_midblock_requested = (not vqdm_scope_sweep and VQDM_TARGET_MODE == "all_midblock")
vqdm_candidate_include_conv = bool(vqdm_scope_sweep or vqdm_include_conv or vqdm_all_midblock_requested)
vqdm_target_layers_all = print_mid_block_quant_layers(
    pipe.unet,
    min_features=16,
    include_linear=True,
    include_conv=vqdm_candidate_include_conv,
    all_linear_conv=vqdm_all_midblock_requested,
)

linear_layers_all = {name: layer for name, layer in vqdm_target_layers_all.items() if isinstance(layer, nn.Linear)}
conv_layers_all = {name: layer for name, layer in vqdm_target_layers_all.items() if isinstance(layer, nn.Conv2d)}

def _vqdm_layer_type_counts(layers: Dict[str, nn.Module]) -> Dict[str, int]:
    return {
        "linear": sum(1 for layer in layers.values() if isinstance(layer, nn.Linear)),
        "conv": sum(1 for layer in layers.values() if isinstance(layer, nn.Conv2d)),
    }


def _select_vqdm_target_layers(target_mode: str, include_conv: bool) -> Dict[str, nn.Module]:
    target_mode = target_mode.strip().lower()
    if not linear_layers_all:
        raise RuntimeError("No Linear target layers found in mid_block")

    if target_mode == "single_largest_linear":
        largest_name = max(linear_layers_all, key=lambda name: linear_layers_all[name].weight.numel())
        target_layers = {largest_name: linear_layers_all[largest_name]}
    elif target_mode == "single_named_linear":
        if vqdm_single_target_layer_name in linear_layers_all:
            target_layers = {vqdm_single_target_layer_name: linear_layers_all[vqdm_single_target_layer_name]}
        else:
            largest_name = max(linear_layers_all, key=lambda name: linear_layers_all[name].weight.numel())
            target_layers = {largest_name: linear_layers_all[largest_name]}
            print(f"Requested target layer not found; selected largest Linear instead: {largest_name}")
    elif target_mode == "all_midblock_linear":
        target_layers = dict(linear_layers_all)
    elif target_mode == "all_midblock":
        if not vqdm_target_layers_all:
            raise RuntimeError("No Linear/Conv2d target layers found in mid_block")
        target_layers = dict(vqdm_target_layers_all)
    elif target_mode == "hardcoded_midblock":
        missing = [name for name in VQDM_HARDCODED_MIDBLOCK_LINEAR_NAMES if name not in vqdm_target_layers_all]
        if missing:
            raise KeyError(f"Hardcoded VQDM target layers were not found: {missing}")
        target_layers = {name: vqdm_target_layers_all[name] for name in VQDM_HARDCODED_MIDBLOCK_LINEAR_NAMES}
    else:
        raise ValueError(
            f"Unsupported VQDM_TARGET_MODE={target_mode!r}; use "
            "single_largest_linear, single_named_linear, all_midblock_linear, all_midblock, hardcoded_midblock"
        )

    if include_conv and target_mode != "all_midblock":
        if not conv_layers_all:
            raise RuntimeError("VQDM Conv scope requested, but no Conv2d target layers were found in mid_block")
        target_layers.update(conv_layers_all)
    return target_layers


def _make_vqdm_scope_spec(
    *,
    key: str,
    label: str,
    target_mode: str,
    include_conv: bool,
    layers: Dict[str, nn.Module],
) -> Dict[str, Any]:
    counts = _vqdm_layer_type_counts(layers)
    return {
        "key": key,
        "label": label,
        "target_mode": target_mode,
        "include_conv": bool(include_conv),
        "layers": dict(layers),
        "layer_names": list(layers.keys()),
        "target_layer_count": len(layers),
        "target_linear_count": counts["linear"],
        "target_conv_count": counts["conv"],
    }


if vqdm_scope_sweep:
    vqdm_scope_specs = [
        _make_vqdm_scope_spec(
            key="single_largest_linear",
            label="single_largest_linear",
            target_mode="single_largest_linear",
            include_conv=False,
            layers=_select_vqdm_target_layers("single_largest_linear", include_conv=False),
        ),
        _make_vqdm_scope_spec(
            key="hardcoded_10_linear",
            label="hardcoded_10_linear",
            target_mode="hardcoded_midblock",
            include_conv=False,
            layers=_select_vqdm_target_layers("hardcoded_midblock", include_conv=False),
        ),
        _make_vqdm_scope_spec(
            key="hardcoded_10_linear_plus_conv",
            label="hardcoded_10_linear_plus_conv",
            target_mode="hardcoded_midblock",
            include_conv=True,
            layers=_select_vqdm_target_layers("hardcoded_midblock", include_conv=True),
        ),
    ]
else:
    legacy_layers = _select_vqdm_target_layers(VQDM_TARGET_MODE, include_conv=vqdm_include_conv)
    legacy_key = VQDM_TARGET_MODE if VQDM_TARGET_MODE == "all_midblock" else VQDM_TARGET_MODE + ("_plus_conv" if vqdm_include_conv else "")
    vqdm_scope_specs = [
        _make_vqdm_scope_spec(
            key=legacy_key,
            label=legacy_key,
            target_mode=VQDM_TARGET_MODE,
            include_conv=vqdm_include_conv,
            layers=legacy_layers,
        )
    ]

print()
print("Selected VQDM target scopes:")
for scope in vqdm_scope_specs:
    print()
    print(
        f"Scope {scope['key']}: {scope['target_layer_count']} layer(s), "
        f"linear={scope['target_linear_count']}, conv={scope['target_conv_count']}, "
        f"target_mode={scope['target_mode']}, include_conv={scope['include_conv']}"
    )
    for name, layer in scope["layers"].items():
        print(f"  {name:60s} {type(layer).__name__} weight={tuple(layer.weight.shape)} params={layer.weight.numel():,}")

vqdm_xtx_layer_name_set = {name for scope in vqdm_scope_specs for name in scope["layer_names"]}
vqdm_xtx_layer_names = [name for name in vqdm_target_layers_all.keys() if name in vqdm_xtx_layer_name_set]
vqdm_xtx_layers = {name: vqdm_target_layers_all[name] for name in vqdm_xtx_layer_names}

# Legacy globals are still used as defaults by helper functions and notebook-style reruns.
vqdm_target_layers = dict(vqdm_xtx_layers)
vqdm_target_layer_names = list(vqdm_xtx_layer_names)

print()
print("VQDM XTX target union:")
print("Target layer count:", len(vqdm_target_layer_names))
print("time_emb_proj included:", any("time_emb_proj" in name for name in vqdm_target_layer_names))
print("target_mode:", VQDM_TARGET_MODE)
print("include_conv:", vqdm_include_conv)
print("scope_sweep:", vqdm_scope_sweep)
print("output_dir:", vqdm_results_dir)
print(f"XTX calibration seed: {vqdm_xtx_seed}")
print(f"Calibration prompts: {len(vqdm_calibration_prompts)} | eval prompts: {len(vqdm_eval_prompts)} | eval seeds: {len(vqdm_eval_seeds_active)}")
print(f"HPS enabled: {vqdm_compute_hps}")

for scope in vqdm_scope_specs:
    print()
    print(f"Compression stats for scope: {scope['key']}")
    scope["compression_by_profile"] = {}
    for profile in vqdm_active_profiles:
        scope["compression_by_profile"][vqdm_profile_key(profile)] = print_vqdm_compression_stats(
            scope["layers"],
            profile,
            make_vqdm_aq_cfg(profile),
        )

vqdm_xtx_signature = {
    "target_layer_names": tuple(vqdm_target_layer_names),
    "scope_keys": tuple(scope["key"] for scope in vqdm_scope_specs),
    "calibration_prompts": tuple(vqdm_calibration_prompts),
    "xtx_steps": int(vqdm_xtx_num_inference_steps),
    "guidance_scale": float(guidance_scale),
    "xtx_seed": int(vqdm_xtx_seed),
    "include_conv": bool(vqdm_candidate_include_conv),
    "scheduler": type(pipe.scheduler).__name__,
    "accum_dtype": str(make_vqdm_aq_cfg(vqdm_active_profiles[0]).accum_dtype),
    "target_mode": VQDM_TARGET_MODE,
    "scope_sweep": bool(vqdm_scope_sweep),
}

reuse_existing_vqdm_xtx = (
    "vqdm_xtx_stats" in globals()
    and "vqdm_xtx_stats_signature" in globals()
    and vqdm_xtx_stats_signature == vqdm_xtx_signature
)

if reuse_existing_vqdm_xtx:
    print("Reusing existing vqdm_xtx_stats from this session.")
    log_cuda_memory("reusing VQDM XTX", vqdm_primary_device)
else:
    log_section("Collect VQDM XTX")
    print("Collecting XTX once for all VQDM bpp profiles...")
    log_event(
        f"calibration_prompts={len(vqdm_calibration_prompts)}, steps={vqdm_xtx_num_inference_steps}, "
        f"collect_devices={[str(d) for d in vqdm_collect_devices]}, xtx_store_device={vqdm_xtx_store_device}, "
        f"seed={vqdm_xtx_seed}"
    )
    log_cuda_memory("before VQDM XTX collection", vqdm_primary_device)
    t0 = time.time()
    vqdm_xtx_stats = collect_midblock_xtx_streaming_autobatch(
        pipe=pipe,
        calibration_prompts=vqdm_calibration_prompts,
        layers=vqdm_target_layers,
        layer_names=vqdm_target_layer_names,
        xtx_batch_size=vqdm_xtx_batch_size,
        num_inference_steps=vqdm_xtx_num_inference_steps,
        guidance_scale=guidance_scale,
        collect_devices=vqdm_collect_devices,
        xtx_store_device=vqdm_xtx_store_device,
        accum_dtype=make_vqdm_aq_cfg(vqdm_active_profiles[0]).accum_dtype,
        group_channels=True,
        show_progress=True,
        generator_seed=vqdm_xtx_seed,
    )
    vqdm_xtx_stats_signature = dict(vqdm_xtx_signature)
    print(f"XTX collected in {time.time() - t0:.1f}s")
    log_cuda_memory("after VQDM XTX collection", vqdm_primary_device)

if not vqdm_keep_eval_pipes_on_device:
    move_pipeline_to_cpu_safely(pipe)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# %% cell 40
# === Shared FP image cache and metrics for VQDM bpp sweep ===

import numpy as np
import torchvision.transforms as T
import open_clip
import lpips


def _module_device(module: nn.Module):
    return next(module.parameters()).device


def vqdm_to_scalar(x):
    if isinstance(x, torch.Tensor):
        return float(x.detach().cpu().flatten()[0].item())
    if isinstance(x, np.ndarray):
        return float(x.reshape(-1)[0])
    if isinstance(x, (list, tuple)):
        return float(x[0])
    return float(x)


print(f"Loading metric models on {vqdm_metric_device}...")
clip_model_vqdm, _, clip_preprocess_vqdm = open_clip.create_model_and_transforms(
    "ViT-H-14",
    pretrained="laion2b_s32b_b79k",
)
clip_model_vqdm = clip_model_vqdm.to(vqdm_metric_device).eval()
clip_tokenizer_vqdm = open_clip.get_tokenizer("ViT-H-14")
lpips_model_vqdm = lpips.LPIPS(net="alex").to(vqdm_metric_device).eval()

lpips_transform_vqdm = T.Compose([
    T.Resize((256, 256)),
    T.CenterCrop((256, 256)),
    T.ToTensor(),
])


def prepare_image_for_clip_vqdm(img):
    return clip_preprocess_vqdm(img).unsqueeze(0).to(_module_device(clip_model_vqdm))


def prepare_image_for_lpips_vqdm(img):
    return lpips_transform_vqdm(img).unsqueeze(0).to(_module_device(lpips_model_vqdm))


@torch.no_grad()
def compute_clip_text_image_score_vqdm(text: str, img_tensor):
    dev = _module_device(clip_model_vqdm)
    txt = clip_tokenizer_vqdm([text]).to(dev)
    txt_emb = clip_model_vqdm.encode_text(txt)
    img_emb = clip_model_vqdm.encode_image(img_tensor.to(dev))
    txt_emb = txt_emb / txt_emb.norm(dim=-1, keepdim=True)
    img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)
    return (txt_emb @ img_emb.T).item()


@torch.no_grad()
def compute_clip_image_image_score_vqdm(img_a, img_b):
    dev = _module_device(clip_model_vqdm)
    e1 = clip_model_vqdm.encode_image(img_a.to(dev))
    e2 = clip_model_vqdm.encode_image(img_b.to(dev))
    e1 = e1 / e1.norm(dim=-1, keepdim=True)
    e2 = e2 / e2.norm(dim=-1, keepdim=True)
    return (e1 @ e2.T).item()


@torch.no_grad()
def compute_lpips_vqdm(img_a, img_b):
    dev = _module_device(lpips_model_vqdm)
    img_a = img_a.to(dev) * 2 - 1
    img_b = img_b.to(dev) * 2 - 1
    return float(lpips_model_vqdm(img_a, img_b).item())


def hps_score_cpu_vqdm(img, prompt, version="v2.1"):
    if not vqdm_compute_hps:
        return float("nan")
    import hpsv2
    orig_is_avail = torch.cuda.is_available
    torch.cuda.is_available = lambda: False
    try:
        return hpsv2.score(img, prompt, hps_version=version)
    finally:
        torch.cuda.is_available = orig_is_avail


def vqdm_fp_cache_key(prompt, seed, steps, guidance_scale, height=None, width=None):
    return (prompt, int(seed), int(steps), float(guidance_scale), height, width)


def count_missing_vqdm_fp_images(cache, prompts, seeds, steps, guidance_scale, height=None, width=None):
    return sum(
        vqdm_fp_cache_key(prompt, seed, steps, guidance_scale, height, width) not in cache
        for prompt in prompts
        for seed in seeds
    )


_VQDM_GEN_BATCH_LOG_CACHE = {}


def resolve_vqdm_generation_batch_size(total_items: Optional[int] = None, label: str = "generation") -> int:
    total_items_int = int(total_items) if total_items is not None else None
    if VQDM_GEN_BATCH_ENV > 0:
        batch = int(VQDM_GEN_BATCH_ENV)
        reason = "env"
    elif not torch.cuda.is_available() or not str(vqdm_eval_device).startswith("cuda"):
        batch = 1
        reason = "cpu"
    else:
        info = _cuda_mem_info(vqdm_eval_device) or {"free_gib": 0.0, "total_gib": 0.0}
        free_gib = float(info.get("free_gib", 0.0))
        total_gib = float(info.get("total_gib", 0.0))
        if total_gib >= 70.0 and free_gib >= 55.0:
            batch = 32
        elif total_gib >= 40.0 and free_gib >= 35.0:
            batch = 24
        elif free_gib >= 24.0:
            batch = 16
        elif free_gib >= 14.0:
            batch = 8
        elif free_gib >= 8.0:
            batch = 4
        else:
            batch = 1
        cap = int(VQDM_GEN_BATCH_MAX) if int(VQDM_GEN_BATCH_MAX) > 0 else (64 if total_gib >= 70.0 else 32)
        batch = min(batch, cap)
        reason = f"auto free={free_gib:.1f}GiB total={total_gib:.1f}GiB cap={cap}"
    if total_items_int is not None and total_items_int > 0:
        batch = min(batch, total_items_int)
    batch = max(1, int(batch))
    cache_key = (label, total_items_int, batch, reason)
    if _VQDM_GEN_BATCH_LOG_CACHE.get(label) != cache_key:
        _VQDM_GEN_BATCH_LOG_CACHE[label] = cache_key
        log_event(f"Resolved generation batch | {label}: batch={batch}, total_items={total_items_int}, reason={reason}")
    return batch


@torch.no_grad()
def generate_images_batched_vqdm(
    pipe_obj,
    prompts_batch,
    seeds_batch,
    *,
    num_inference_steps,
    guidance_scale,
    height=None,
    width=None,
    device=None,
    dtype=None,
    keep_on_device=None,
):
    prompts_batch = list(prompts_batch)
    seeds_batch = [int(s) for s in seeds_batch]
    if len(prompts_batch) != len(seeds_batch):
        raise ValueError("prompts_batch and seeds_batch must have the same length")
    if not prompts_batch:
        return []
    device = device or vqdm_eval_device
    dtype = dtype or (runtime_dtype if str(device).startswith("cuda") else torch.float32)
    keep_on_device = vqdm_keep_eval_pipes_on_device if keep_on_device is None else keep_on_device
    gen_device = device if str(device).startswith("cuda") else "cpu"
    if len(prompts_batch) == 1:
        img = generate_one_image(
            pipe_obj,
            prompts_batch[0],
            seed=seeds_batch[0],
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            height=height,
            width=width,
            device=device,
            dtype=dtype,
        )
        if not keep_on_device:
            move_pipeline_to_cpu_safely(pipe_obj)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        return [img]

    try:
        pipe_obj = force_pipeline_device_dtype(pipe_obj, device, dtype)
        pipe_obj.set_progress_bar_config(disable=True)
        generators = [torch.Generator(device=gen_device).manual_seed(s) for s in seeds_batch]
        out = pipe_obj(
            prompts_batch,
            generator=generators,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            height=height,
            width=width,
        )
        images = list(out.images)
        if not keep_on_device:
            move_pipeline_to_cpu_safely(pipe_obj)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        return images
    except BaseException as exc:
        if _is_cuda_oom(exc) and len(prompts_batch) > 1:
            _clear_cuda_after_oom()
            mid = len(prompts_batch) // 2
            log_event(
                f"CUDA OOM in batched generation; splitting batch {len(prompts_batch)} -> "
                f"{mid}+{len(prompts_batch) - mid}"
            )
            images = (
                generate_images_batched_vqdm(
                    pipe_obj,
                    prompts_batch[:mid],
                    seeds_batch[:mid],
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    height=height,
                    width=width,
                    device=device,
                    dtype=dtype,
                    keep_on_device=True,
                )
                + generate_images_batched_vqdm(
                    pipe_obj,
                    prompts_batch[mid:],
                    seeds_batch[mid:],
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    height=height,
                    width=width,
                    device=device,
                    dtype=dtype,
                    keep_on_device=True,
                )
            )
            if not keep_on_device:
                move_pipeline_to_cpu_safely(pipe_obj)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            return images
        raise


def prepare_images_for_clip_vqdm(imgs):
    return torch.stack([clip_preprocess_vqdm(img) for img in imgs], dim=0).to(_module_device(clip_model_vqdm))


def prepare_images_for_lpips_vqdm(imgs):
    return torch.stack([lpips_transform_vqdm(img) for img in imgs], dim=0).to(_module_device(lpips_model_vqdm))


@torch.no_grad()
def compute_pixel_metrics_batch_vqdm(img_fp_pils, img_test_pils):
    fp = torch.stack([T.ToTensor()(img.convert("RGB")) for img in img_fp_pils], dim=0).float()
    test = torch.stack([T.ToTensor()(img.convert("RGB")) for img in img_test_pils], dim=0).float()
    diff = test - fp
    mse = diff.square().flatten(1).mean(dim=1)
    mae = diff.abs().flatten(1).mean(dim=1)
    psnr = -10.0 * torch.log10(mse.clamp_min(1e-12))

    dims = (1, 2, 3)
    mu_x = fp.mean(dim=dims)
    mu_y = test.mean(dim=dims)
    var_x = ((fp - mu_x[:, None, None, None]) ** 2).mean(dim=dims)
    var_y = ((test - mu_y[:, None, None, None]) ** 2).mean(dim=dims)
    cov_xy = ((fp - mu_x[:, None, None, None]) * (test - mu_y[:, None, None, None])).mean(dim=dims)
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    ssim = ((2 * mu_x * mu_y + c1) * (2 * cov_xy + c2)) / (
        (mu_x.square() + mu_y.square() + c1) * (var_x + var_y + c2)
    )

    return [
        {
            "pixel_mse": float(mse[i].item()),
            "pixel_mae": float(mae[i].item()),
            "psnr": float(psnr[i].item()),
            "ssim": float(ssim[i].item()),
        }
        for i in range(fp.shape[0])
    ]


@torch.no_grad()
def compute_metrics_batch_against_fp_vqdm(img_fp_pils, img_test_pils, prompts_batch):
    dev_clip = _module_device(clip_model_vqdm)
    prompts_batch = list(prompts_batch)

    img_fp_clip = prepare_images_for_clip_vqdm(img_fp_pils)
    img_q_clip = prepare_images_for_clip_vqdm(img_test_pils)
    txt = clip_tokenizer_vqdm(prompts_batch).to(dev_clip)

    txt_emb = clip_model_vqdm.encode_text(txt)
    fp_emb = clip_model_vqdm.encode_image(img_fp_clip.to(dev_clip))
    q_emb = clip_model_vqdm.encode_image(img_q_clip.to(dev_clip))
    txt_emb = txt_emb / txt_emb.norm(dim=-1, keepdim=True)
    fp_emb = fp_emb / fp_emb.norm(dim=-1, keepdim=True)
    q_emb = q_emb / q_emb.norm(dim=-1, keepdim=True)

    clip_ti_fp = (txt_emb * fp_emb).sum(dim=-1).detach().cpu().tolist()
    clip_ti = (txt_emb * q_emb).sum(dim=-1).detach().cpu().tolist()
    clip_ii = (fp_emb * q_emb).sum(dim=-1).detach().cpu().tolist()

    img_fp_lp = prepare_images_for_lpips_vqdm(img_fp_pils).to(_module_device(lpips_model_vqdm)) * 2 - 1
    img_q_lp = prepare_images_for_lpips_vqdm(img_test_pils).to(_module_device(lpips_model_vqdm)) * 2 - 1
    lpips_vals = lpips_model_vqdm(img_fp_lp, img_q_lp).flatten().detach().cpu().tolist()
    pix = compute_pixel_metrics_batch_vqdm(img_fp_pils, img_test_pils)

    return [
        {
            "lpips": float(lpips_vals[i]),
            "clip_ti_fp": float(clip_ti_fp[i]),
            "clip_ti": float(clip_ti[i]),
            "delta_clip_ti": float(clip_ti[i] - clip_ti_fp[i]),
            "clip_ii": float(clip_ii[i]),
            **pix[i],
        }
        for i in range(len(prompts_batch))
    ]


VQDM_FORWARD_METRIC_PROMPTS = _env_int("VQDM_FORWARD_METRIC_PROMPTS", 2)
VQDM_FORWARD_METRIC_TIMESTEPS = _env_int("VQDM_FORWARD_METRIC_TIMESTEPS", 3)
VQDM_FORWARD_METRIC_SEED = _env_int("VQDM_FORWARD_METRIC_SEED", 20260515)


def _vqdm_forward_metric_timesteps(n_steps: int) -> List[int]:
    n_steps = max(1, int(n_steps))
    scheduler_config = getattr(getattr(pipe, "scheduler", None), "config", None)
    total = int(getattr(scheduler_config, "num_train_timesteps", 1000))
    hi = max(1, total - max(1, total // 10))
    lo = max(1, total // 10)
    if n_steps == 1:
        return [int((hi + lo) // 2)]
    return [int(round(x)) for x in np.linspace(hi, lo, n_steps)]


def _vqdm_forward_metric_prompts(prompts: Sequence[str]) -> List[str]:
    prompts_list = [str(p) for p in prompts]
    limit = max(1, int(VQDM_FORWARD_METRIC_PROMPTS))
    return prompts_list[:limit] if prompts_list else [""]


def _vqdm_compare_tensors(prefix: str, fp: torch.Tensor, q: torch.Tensor) -> Dict[str, float]:
    fp_f = fp.detach().float().reshape(fp.shape[0], -1)
    q_f = q.detach().float().reshape(q.shape[0], -1)
    diff = q_f - fp_f
    mse_per = diff.square().mean(dim=1)
    fp_energy_per = fp_f.square().mean(dim=1).clamp_min(1e-12)
    rel_mse_per = mse_per / fp_energy_per
    rel_l2_per = diff.norm(dim=1) / fp_f.norm(dim=1).clamp_min(1e-12)
    cosine_per = F.cosine_similarity(fp_f, q_f, dim=1)
    return {
        f"{prefix}_mse": float(mse_per.mean().item()),
        f"{prefix}_rel_mse": float(rel_mse_per.mean().item()),
        f"{prefix}_rel_l2": float(rel_l2_per.mean().item()),
        f"{prefix}_cosine": float(cosine_per.mean().item()),
        f"{prefix}_max_abs": float(diff.abs().max().item()),
    }


@torch.no_grad()
def _vqdm_make_forward_probe_inputs(pipe_obj, prompts: Sequence[str]):
    device = torch.device(vqdm_eval_device if torch.cuda.is_available() else "cpu")
    dtype = vqdm_final_dtype if device.type == "cuda" else torch.float32
    probe_prompts = _vqdm_forward_metric_prompts(prompts)

    pipe_obj.unet.to(device, dtype=dtype)
    pipe_obj.text_encoder.to(device, dtype=dtype)
    text_inputs = pipe_obj.tokenizer(
        probe_prompts,
        padding="max_length",
        max_length=pipe_obj.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    input_ids = text_inputs.input_ids.to(device)
    encoder_hidden_states = pipe_obj.text_encoder(input_ids)[0].to(device=device, dtype=dtype)

    sample_size = getattr(pipe_obj.unet.config, "sample_size", 64)
    if isinstance(sample_size, (tuple, list)):
        height, width = int(sample_size[0]), int(sample_size[1])
    else:
        height = width = int(sample_size)
    generator = torch.Generator(device=device).manual_seed(int(VQDM_FORWARD_METRIC_SEED))
    sample = torch.randn(
        len(probe_prompts),
        int(pipe_obj.unet.config.in_channels),
        height,
        width,
        generator=generator,
        device=device,
        dtype=dtype,
    )
    timesteps = _vqdm_forward_metric_timesteps(VQDM_FORWARD_METRIC_TIMESTEPS)
    return sample, timesteps, encoder_hidden_states


@torch.no_grad()
def _vqdm_run_unet_forward_probe(pipe_obj, sample: torch.Tensor, timesteps: Sequence[int], encoder_hidden_states: torch.Tensor):
    outputs: List[torch.Tensor] = []
    mid_outputs: List[torch.Tensor] = []

    def _hook(_module, _inputs, output):
        out = output[0] if isinstance(output, (tuple, list)) else output
        mid_outputs.append(out.detach().float().cpu())

    handle = pipe_obj.unet.mid_block.register_forward_hook(_hook)
    try:
        for timestep in timesteps:
            t = torch.full((sample.shape[0],), int(timestep), device=sample.device, dtype=torch.long)
            out = pipe_obj.unet(sample, t, encoder_hidden_states=encoder_hidden_states).sample
            outputs.append(out.detach().float().cpu())
    finally:
        handle.remove()

    return torch.cat(outputs, dim=0), torch.cat(mid_outputs, dim=0)


@torch.no_grad()
def compute_forward_drift_between_vqdm_pipes(pipe_fp, pipe_test, prompts: Sequence[str], eval_label: str = "VQDM") -> Dict[str, float]:
    if int(VQDM_FORWARD_METRIC_PROMPTS) <= 0 or int(VQDM_FORWARD_METRIC_TIMESTEPS) <= 0:
        return {}
    try:
        sample, timesteps, encoder_hidden_states = _vqdm_make_forward_probe_inputs(pipe_fp, prompts)
        pipe_test.unet.to(sample.device, dtype=sample.dtype)
        fp_out, fp_mid = _vqdm_run_unet_forward_probe(pipe_fp, sample, timesteps, encoder_hidden_states)
        q_out, q_mid = _vqdm_run_unet_forward_probe(pipe_test, sample, timesteps, encoder_hidden_states)
        metrics = {}
        metrics.update(_vqdm_compare_tensors("unet_forward", fp_out, q_out))
        metrics.update(_vqdm_compare_tensors("midblock_forward", fp_mid, q_mid))
        log_event(
            f"Forward drift | {eval_label} | "
            f"unet_rel_l2={metrics['unet_forward_rel_l2']:.6f}, "
            f"mid_rel_l2={metrics['midblock_forward_rel_l2']:.6f}, "
            f"prompts={sample.shape[0]}, timesteps={len(timesteps)}"
        )
        return metrics
    except Exception as exc:
        log_event(f"Forward drift skipped for {eval_label}: {type(exc).__name__}: {exc}")
        return {}


@torch.no_grad()
def compute_forward_drift_with_vqdm_midblock_patch(
    pipe_obj,
    mid_block_q: nn.Module,
    prompts: Sequence[str],
    eval_label: str = "VQDM",
) -> Dict[str, float]:
    if int(VQDM_FORWARD_METRIC_PROMPTS) <= 0 or int(VQDM_FORWARD_METRIC_TIMESTEPS) <= 0:
        return {}
    fp_mid_block = pipe_obj.unet.mid_block
    fp_was_training = pipe_obj.unet.training
    try:
        sample, timesteps, encoder_hidden_states = _vqdm_make_forward_probe_inputs(pipe_obj, prompts)
        fp_out, fp_mid = _vqdm_run_unet_forward_probe(pipe_obj, sample, timesteps, encoder_hidden_states)
        pipe_obj.unet.mid_block = mid_block_q.to(sample.device, dtype=sample.dtype)
        pipe_obj.unet.eval()
        q_out, q_mid = _vqdm_run_unet_forward_probe(pipe_obj, sample, timesteps, encoder_hidden_states)
        metrics = {}
        metrics.update(_vqdm_compare_tensors("unet_forward", fp_out, q_out))
        metrics.update(_vqdm_compare_tensors("midblock_forward", fp_mid, q_mid))
        log_event(
            f"Forward drift | {eval_label} in-place | "
            f"unet_rel_l2={metrics['unet_forward_rel_l2']:.6f}, "
            f"mid_rel_l2={metrics['midblock_forward_rel_l2']:.6f}, "
            f"prompts={sample.shape[0]}, timesteps={len(timesteps)}"
        )
        return metrics
    except Exception as exc:
        log_event(f"Forward drift skipped for {eval_label}: {type(exc).__name__}: {exc}")
        return {}
    finally:
        pipe_obj.unet.mid_block = fp_mid_block
        if fp_was_training:
            pipe_obj.unet.train()
        else:
            pipe_obj.unet.eval()


@torch.no_grad()
def precompute_vqdm_fp_image_cache(
    pipe_fp,
    prompts,
    seeds,
    cache,
    image_pbar=None,
    steps=None,
    guidance_scale=None,
    height=None,
    width=None,
):
    steps = vqdm_eval_num_inference_steps if steps is None else int(steps)
    guidance_scale = vqdm_eval_guidance_scale if guidance_scale is None else float(guidance_scale)
    keep_after_batch = vqdm_keep_eval_pipes_on_device
    try:
        pending = []
        for prompt in prompts:
            for seed in seeds:
                key = vqdm_fp_cache_key(prompt, seed, steps, guidance_scale, height, width)
                if key not in cache:
                    pending.append((prompt, seed, key))

        gen_batch = resolve_vqdm_generation_batch_size(len(pending), label="VQDM FP cache")
        for start in range(0, len(pending), gen_batch):
            chunk = pending[start:start + gen_batch]
            if not chunk:
                continue
            chunk_prompts = [x[0] for x in chunk]
            chunk_seeds = [x[1] for x in chunk]
            if image_pbar is not None:
                image_pbar.set_postfix_str(f"FP batch={len(chunk)}, seed={chunk_seeds[0]}")
            imgs = generate_images_batched_vqdm(
                pipe_fp,
                chunk_prompts,
                chunk_seeds,
                num_inference_steps=steps,
                guidance_scale=guidance_scale,
                height=height,
                width=width,
                device=vqdm_eval_device,
                keep_on_device=True,
            )
            for (_, _, key), img in zip(chunk, imgs):
                cache[key] = img
            if image_pbar is not None:
                image_pbar.update(len(chunk))
    finally:
        if not keep_after_batch:
            move_pipeline_to_cpu_safely(pipe_fp)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


@torch.no_grad()
def eval_vqdm_prompts_against_fp(
    pipe_fp,
    pipe_test,
    prompts,
    seeds,
    fp_cache=None,
    image_pbar=None,
    eval_label="VQDM",
    forward_drift_metrics: Optional[Dict[str, float]] = None,
    skip_forward_drift: bool = False,
):
    rows = []
    fp_cache = {} if fp_cache is None else fp_cache
    pairs = [(prompt, seed) for prompt in prompts for seed in seeds]
    keep_after_batch = vqdm_keep_eval_pipes_on_device
    forward_drift_metrics = dict(forward_drift_metrics or {})

    precompute_vqdm_fp_image_cache(
        pipe_fp,
        prompts,
        seeds,
        cache=fp_cache,
        image_pbar=image_pbar,
        steps=vqdm_eval_num_inference_steps,
        guidance_scale=vqdm_eval_guidance_scale,
    )

    try:
        gen_batch = resolve_vqdm_generation_batch_size(len(pairs), label=eval_label)
        for start in range(0, len(pairs), gen_batch):
            chunk = pairs[start:start + gen_batch]
            chunk_prompts = [p for p, _ in chunk]
            chunk_seeds = [s for _, s in chunk]
            if image_pbar is not None and chunk:
                image_pbar.set_postfix_str(f"{eval_label} batch={len(chunk)}, seed={chunk_seeds[0]}")

            imgs_fp = [
                fp_cache[vqdm_fp_cache_key(p, s, vqdm_eval_num_inference_steps, vqdm_eval_guidance_scale)]
                for p, s in chunk
            ]

            t_gen0 = time.perf_counter()
            imgs_q = generate_images_batched_vqdm(
                pipe_test,
                chunk_prompts,
                chunk_seeds,
                num_inference_steps=vqdm_eval_num_inference_steps,
                guidance_scale=vqdm_eval_guidance_scale,
                device=vqdm_eval_device,
                keep_on_device=True,
            )
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t_gen = time.perf_counter() - t_gen0
            if image_pbar is not None:
                image_pbar.update(len(chunk))
                image_pbar.set_postfix_str(f"{eval_label} metrics batch={len(chunk)}")

            t_met0 = time.perf_counter()
            metric_rows = compute_metrics_batch_against_fp_vqdm(imgs_fp, imgs_q, chunk_prompts)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t_met = time.perf_counter() - t_met0
            log_event(
                f"Eval batch | {eval_label} | n={len(chunk)}, gen={t_gen:.2f}s, metrics={t_met:.2f}s, "
                f"gen_batch={gen_batch}"
            )

            for i, ((prompt, seed), metrics) in enumerate(zip(chunk, metric_rows)):
                if vqdm_compute_hps:
                    try:
                        hps_fp = vqdm_to_scalar(hps_score_cpu_vqdm(imgs_fp[i], prompt))
                        hps_q = vqdm_to_scalar(hps_score_cpu_vqdm(imgs_q[i], prompt))
                    except Exception as exc:
                        print(f"HPS skipped for this pair: {type(exc).__name__}: {exc}")
                        hps_fp = float("nan")
                        hps_q = float("nan")
                else:
                    hps_fp = float("nan")
                    hps_q = float("nan")
                rows.append({
                    "prompt": prompt,
                    "seed": seed,
                    "lpips": metrics["lpips"],
                    "clip_ti_fp": metrics["clip_ti_fp"],
                    "clip_ti": metrics["clip_ti"],
                    "delta_clip_ti": metrics["delta_clip_ti"],
                    "clip_ii": metrics["clip_ii"],
                    "pixel_mse": metrics["pixel_mse"],
                    "pixel_mae": metrics["pixel_mae"],
                    "psnr": metrics["psnr"],
                    "ssim": metrics["ssim"],
                    "hps": hps_q,
                    "hps_fp": hps_fp,
                })

        if not skip_forward_drift and not forward_drift_metrics:
            forward_drift_metrics = compute_forward_drift_between_vqdm_pipes(
                pipe_fp,
                pipe_test,
                prompts,
                eval_label=eval_label,
            )
    finally:
        if not keep_after_batch:
            move_pipeline_to_cpu_safely(pipe_test)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def mean_finite(key):
        vals = [r[key] for r in rows if np.isfinite(r[key])]
        return sum(vals) / len(vals) if vals else float("nan")

    agg = {
        "lpips": mean_finite("lpips"),
        "clip_ti_fp": mean_finite("clip_ti_fp"),
        "clip_ti": mean_finite("clip_ti"),
        "delta_clip_ti": mean_finite("delta_clip_ti"),
        "clip_ii": mean_finite("clip_ii"),
        "pixel_mse": mean_finite("pixel_mse"),
        "pixel_mae": mean_finite("pixel_mae"),
        "psnr": mean_finite("psnr"),
        "ssim": mean_finite("ssim"),
        "hps": mean_finite("hps"),
        "hps_fp": mean_finite("hps_fp"),
        "n": len(rows),
    }
    agg.update(forward_drift_metrics)
    return agg, rows


@torch.no_grad()
def eval_with_temporary_vqdm_midblock(
    *,
    pipe_obj,
    mid_block_q: nn.Module,
    prompts,
    seeds,
    fp_cache,
    image_pbar=None,
    eval_label: str = "VQDM",
):
    """Evaluate a quantized VQDM mid_block without deepcopying the whole pipeline."""
    missing = count_missing_vqdm_fp_images(
        fp_cache,
        prompts,
        seeds,
        steps=vqdm_eval_num_inference_steps,
        guidance_scale=vqdm_eval_guidance_scale,
    )
    if missing:
        log_event(f"FP cache missing {missing} image(s) before in-place eval; precomputing before patch")
        precompute_vqdm_fp_image_cache(
            pipe_obj,
            prompts,
            seeds,
            cache=fp_cache,
            image_pbar=image_pbar,
            steps=vqdm_eval_num_inference_steps,
            guidance_scale=vqdm_eval_guidance_scale,
        )

    fp_mid_block = pipe_obj.unet.mid_block
    fp_was_training = pipe_obj.unet.training
    forward_drift_metrics = compute_forward_drift_with_vqdm_midblock_patch(
        pipe_obj,
        mid_block_q,
        prompts,
        eval_label=eval_label,
    )
    try:
        pipe_obj.unet.mid_block = mid_block_q.to(vqdm_eval_device, dtype=vqdm_final_dtype)
        optimize_pipeline_for_inference(pipe_obj, label=f"{eval_label} in-place patched pipeline")
        log_event(f"Evaluating {eval_label} by in-place mid_block patch; full pipeline deepcopy skipped")
        log_cuda_memory(f"after in-place mid_block patch for {eval_label}", vqdm_primary_device)
        return eval_vqdm_prompts_against_fp(
            pipe_obj,
            pipe_obj,
            prompts,
            seeds,
            fp_cache=fp_cache,
            image_pbar=image_pbar,
            eval_label=eval_label,
            forward_drift_metrics=forward_drift_metrics,
            skip_forward_drift=True,
        )
    finally:
        pipe_obj.unet.mid_block = fp_mid_block
        if fp_was_training:
            pipe_obj.unet.train()
        else:
            pipe_obj.unet.eval()
        optimize_pipeline_for_inference(pipe_obj, label="FP pipeline restored after in-place eval")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        log_event(f"Restored FP mid_block after {eval_label}")
        log_cuda_memory(f"after restoring FP mid_block for {eval_label}", vqdm_primary_device)


vqdm_fp_image_cache = globals().get("vqdm_fp_image_cache", {})
vqdm_missing_fp = count_missing_vqdm_fp_images(
    vqdm_fp_image_cache,
    vqdm_eval_prompts,
    vqdm_eval_seeds_active,
    steps=vqdm_eval_num_inference_steps,
    guidance_scale=vqdm_eval_guidance_scale,
)

log_section("Prepare VQDM FP image cache")
print(f"Precomputing/reusing VQDM FP images: missing={vqdm_missing_fp}, cache_total_before={len(vqdm_fp_image_cache)}")
log_event(
    f"cache_prompts={len(vqdm_eval_prompts)}, cache_seeds={len(vqdm_eval_seeds_active)}, "
    f"steps={vqdm_eval_num_inference_steps}, guidance_scale={vqdm_eval_guidance_scale}"
)
log_cuda_memory("before VQDM FP image cache", vqdm_primary_device)
if vqdm_missing_fp:
    with tqdm(total=vqdm_missing_fp, desc="VQDM FP image cache", unit="image") as image_pbar:
        precompute_vqdm_fp_image_cache(
            pipe,
            vqdm_eval_prompts,
            vqdm_eval_seeds_active,
            cache=vqdm_fp_image_cache,
            image_pbar=image_pbar,
            steps=vqdm_eval_num_inference_steps,
            guidance_scale=vqdm_eval_guidance_scale,
        )
print(f"FP cache total after precompute: {len(vqdm_fp_image_cache)}")
log_cuda_memory("after VQDM FP image cache", vqdm_primary_device)


def compute_vqdm_code_usage_stats(quantized_weight: "QuantizedWeight") -> Dict[str, Any]:
    codes = quantized_weight.get_codes().detach().to("cpu", dtype=torch.long)
    if codes.ndim != 3:
        codes = codes.reshape(-1, 1, int(quantized_weight.num_codebooks))
    num_codebooks = int(quantized_weight.num_codebooks)
    K = int(quantized_weight.codebook_size)
    bits = int(quantized_weight.nbits_per_codebook)
    group_size = int(quantized_weight.out_group_size) * int(quantized_weight.in_group_size)

    per_stage = []
    total_entropy = 0.0
    total_used = 0
    total_dead = 0
    total_entries = 0
    invalid_codes = 0
    for m in range(num_codebooks):
        flat = codes[..., m].reshape(-1)
        invalid = int(((flat < 0) | (flat >= K)).sum().item())
        invalid_codes += invalid
        flat_valid = flat[(flat >= 0) & (flat < K)]
        counts = torch.bincount(flat_valid, minlength=K).to(torch.float64)
        n = float(counts.sum().item())
        used = int((counts > 0).sum().item())
        probs = counts[counts > 0] / max(n, 1.0)
        entropy = float((-(probs * torch.log2(probs))).sum().item()) if probs.numel() else 0.0
        per_stage.append({
            "codebook": int(m),
            "K": int(K),
            "bits": int(bits),
            "n_codes": int(n),
            "used_codes": int(used),
            "dead_codes": int(K - used),
            "utilization": float(used / K),
            "entropy": entropy,
            "perplexity": float(2.0 ** entropy),
            "normalized_entropy": float(entropy / bits) if bits > 0 else 0.0,
            "max_code_probability": float((counts.max() / max(n, 1.0)).item()) if counts.numel() else 0.0,
            "invalid_codes": int(invalid),
        })
        total_entropy += entropy
        total_used += used
        total_dead += K - used
        total_entries += K

    packed_bits = num_codebooks * bits
    return {
        "num_codebooks": int(num_codebooks),
        "K": int(K),
        "bits": int(bits),
        "group_size": int(group_size),
        "packed_index_bpp": float(packed_bits / max(group_size, 1)),
        "entropy_bpp": float(total_entropy / max(group_size, 1)),
        "entropy_bits_per_vector": float(total_entropy),
        "packed_bits_per_vector": int(packed_bits),
        "used_codes_total": int(total_used),
        "dead_codes_total": int(total_dead),
        "codebook_entries_total": int(total_entries),
        "mean_utilization": float(total_used / total_entries) if total_entries else 0.0,
        "mean_normalized_entropy": float(sum(s["normalized_entropy"] for s in per_stage) / len(per_stage)) if per_stage else 0.0,
        "invalid_codes": int(invalid_codes),
        "per_stage": per_stage,
    }


def collect_vqdm_layer_diagnostics(module: nn.Module, *, prefix: str = "mid_block") -> List[Dict[str, Any]]:
    diagnostics: List[Dict[str, Any]] = []
    for name, child in module.named_modules():
        quantized_weight = getattr(child, "quantized_weight", None)
        if quantized_weight is None or not hasattr(quantized_weight, "get_codes"):
            continue
        full_name = f"{prefix}.{name}" if prefix and name else prefix
        usage = compute_vqdm_code_usage_stats(quantized_weight)
        weight_params = int(quantized_weight.out_features) * int(quantized_weight.in_features)
        diagnostics.append({
            "name": full_name,
            "module_type": type(child).__name__,
            "weight_params": int(weight_params),
            "num_codebooks": int(quantized_weight.num_codebooks),
            "K": int(quantized_weight.codebook_size),
            "bits": int(quantized_weight.nbits_per_codebook),
            "in_group_size": int(quantized_weight.in_group_size),
            "out_group_size": int(quantized_weight.out_group_size),
            "code_usage": usage,
        })
    return diagnostics


def summarize_vqdm_code_usage_from_diagnostics(diagnostics: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    total_weight = 0
    weighted_entropy_bpp = 0.0
    weighted_packed_bpp = 0.0
    weighted_utilization = 0.0
    weighted_norm_entropy = 0.0
    total_used = 0
    total_dead = 0
    total_entries = 0
    invalid = 0
    layers_with_stats = 0
    for diag in diagnostics:
        usage = diag.get("code_usage", {}) or {}
        weight = int(diag.get("weight_params", 0) or 0)
        if not usage or weight <= 0:
            continue
        layers_with_stats += 1
        total_weight += weight
        weighted_entropy_bpp += weight * float(usage.get("entropy_bpp", 0.0))
        weighted_packed_bpp += weight * float(usage.get("packed_index_bpp", 0.0))
        weighted_utilization += weight * float(usage.get("mean_utilization", 0.0))
        weighted_norm_entropy += weight * float(usage.get("mean_normalized_entropy", 0.0))
        total_used += int(usage.get("used_codes_total", 0) or 0)
        total_dead += int(usage.get("dead_codes_total", 0) or 0)
        total_entries += int(usage.get("codebook_entries_total", 0) or 0)
        invalid += int(usage.get("invalid_codes", 0) or 0)

    denom = max(total_weight, 1)
    return {
        "layers_with_stats": int(layers_with_stats),
        "weighted_entropy_bpp": float(weighted_entropy_bpp / denom),
        "weighted_packed_index_bpp": float(weighted_packed_bpp / denom),
        "weighted_mean_utilization": float(weighted_utilization / denom),
        "weighted_mean_normalized_entropy": float(weighted_norm_entropy / denom),
        "used_codes_total": int(total_used),
        "dead_codes_total": int(total_dead),
        "codebook_entries_total": int(total_entries),
        "global_utilization": float(total_used / total_entries) if total_entries else 0.0,
        "invalid_codes": int(invalid),
    }


def _json_ready(value):
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
    if isinstance(value, np.ndarray):
        return _json_ready(value.tolist())
    if isinstance(value, np.generic):
        return _json_ready(value.item())
    if isinstance(value, torch.Tensor):
        return _json_ready(value.detach().cpu().tolist())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def _vqdm_scope_summary(scope: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "key": scope["key"],
        "label": scope["label"],
        "target_mode": scope["target_mode"],
        "include_conv": bool(scope["include_conv"]),
        "target_layer_count": int(scope["target_layer_count"]),
        "target_linear_count": int(scope["target_linear_count"]),
        "target_conv_count": int(scope["target_conv_count"]),
        "layer_names": list(scope["layer_names"]),
        "compression_by_profile": scope.get("compression_by_profile", {}),
    }


def save_vqdm_sweep_outputs(results: Sequence[Dict[str, Any]]):
    sorted_by_score = sorted(results, key=lambda r: r.get("score", float("-inf")), reverse=True)
    payload = {
        "settings": {
            "runner": "vqdm-sd1-5-optimized.py",
            "run_id": VQDM_RUN_ID,
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "scope_sweep": bool(vqdm_scope_sweep),
            "target_mode_env": VQDM_TARGET_MODE,
            "include_conv_env": bool(vqdm_include_conv),
            "output_dir": str(vqdm_results_dir),
            "calibration_prompts": len(vqdm_calibration_prompts),
            "eval_prompts": len(vqdm_eval_prompts),
            "eval_seeds": list(vqdm_eval_seeds_active),
            "eval_items": len(vqdm_eval_prompts) * len(vqdm_eval_seeds_active),
            "num_inference_steps": int(vqdm_eval_num_inference_steps),
            "guidance_scale": float(vqdm_eval_guidance_scale),
            "xtx_steps": int(vqdm_xtx_num_inference_steps),
            "xtx_seed": int(vqdm_xtx_seed),
            "profiles": [dict(p) for p in vqdm_active_profiles],
            "scopes": [_vqdm_scope_summary(scope) for scope in vqdm_scope_specs],
            "devices": [str(d) for d in vqdm_devices],
            "collect_devices": [str(d) for d in vqdm_collect_devices],
            "quant_devices": [str(d) for d in vqdm_quant_devices],
            "metric_device": str(vqdm_metric_device),
            "compute_hps": bool(vqdm_compute_hps),
        },
        "summaries": {
            "final_top": sorted_by_score[:20],
            "best_by_scope": [
                max(
                    [r for r in results if r["scope_key"] == scope["key"]],
                    key=lambda r: r.get("score", float("-inf")),
                )
                for scope in vqdm_scope_specs
                if any(r["scope_key"] == scope["key"] for r in results)
            ],
        },
        "results": list(results),
    }

    tmp_json = vqdm_results_json_path.with_suffix(vqdm_results_json_path.suffix + ".tmp")
    with tmp_json.open("w", encoding="utf-8") as f:
        json.dump(_json_ready(payload), f, ensure_ascii=False, indent=2)
    os.replace(tmp_json, vqdm_results_json_path)

    fields = [
        "result_index",
        "scope_key",
        "scope_label",
        "target_mode",
        "include_conv",
        "target_layer_count",
        "target_linear_count",
        "target_conv_count",
        "profile_key",
        "profile_label",
        "reference_bpp",
        "estimated_bpp",
        "estimated_compression",
        "num_codebooks",
        "nbits_per_codebook",
        "codebook_size",
        "entropy_bpp",
        "packed_index_bpp",
        "mean_utilization",
        "mean_normalized_entropy",
        "score",
        "lpips",
        "clip_ii",
        "clip_ti",
        "clip_ti_fp",
        "delta_clip_ti",
        "pixel_mse",
        "pixel_mae",
        "psnr",
        "ssim",
        "unet_forward_mse",
        "unet_forward_rel_mse",
        "unet_forward_rel_l2",
        "unet_forward_cosine",
        "unet_forward_max_abs",
        "midblock_forward_mse",
        "midblock_forward_rel_mse",
        "midblock_forward_rel_l2",
        "midblock_forward_cosine",
        "midblock_forward_max_abs",
        "hps",
        "hps_fp",
        "n",
        "quant_build_s",
        "eval_s",
        "total_s",
    ]
    tmp_csv = vqdm_results_csv_path.with_suffix(vqdm_results_csv_path.suffix + ".tmp")
    with tmp_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for idx, r in enumerate(results):
            agg = r["agg"]
            compression = r.get("compression", {})
            timing = r.get("timing", {})
            usage = r.get("code_usage_summary", {}) or {}
            writer.writerow({
                "result_index": idx,
                "scope_key": r.get("scope_key"),
                "scope_label": r.get("scope_label"),
                "target_mode": r.get("target_mode"),
                "include_conv": r.get("include_conv"),
                "target_layer_count": r.get("target_layer_count"),
                "target_linear_count": r.get("target_linear_count"),
                "target_conv_count": r.get("target_conv_count"),
                "profile_key": r.get("profile_key"),
                "profile_label": r.get("profile_label"),
                "reference_bpp": r.get("reference_bpp"),
                "estimated_bpp": compression.get("estimated_bpp"),
                "estimated_compression": compression.get("estimated_compression"),
                "num_codebooks": r.get("num_codebooks"),
                "nbits_per_codebook": r.get("nbits_per_codebook"),
                "codebook_size": r.get("codebook_size"),
                "entropy_bpp": usage.get("weighted_entropy_bpp"),
                "packed_index_bpp": usage.get("weighted_packed_index_bpp"),
                "mean_utilization": usage.get("weighted_mean_utilization"),
                "mean_normalized_entropy": usage.get("weighted_mean_normalized_entropy"),
                "score": r.get("score"),
                "lpips": agg.get("lpips"),
                "clip_ii": agg.get("clip_ii"),
                "clip_ti": agg.get("clip_ti"),
                "clip_ti_fp": agg.get("clip_ti_fp"),
                "delta_clip_ti": agg.get("delta_clip_ti"),
                "pixel_mse": agg.get("pixel_mse"),
                "pixel_mae": agg.get("pixel_mae"),
                "psnr": agg.get("psnr"),
                "ssim": agg.get("ssim"),
                "unet_forward_mse": agg.get("unet_forward_mse"),
                "unet_forward_rel_mse": agg.get("unet_forward_rel_mse"),
                "unet_forward_rel_l2": agg.get("unet_forward_rel_l2"),
                "unet_forward_cosine": agg.get("unet_forward_cosine"),
                "unet_forward_max_abs": agg.get("unet_forward_max_abs"),
                "midblock_forward_mse": agg.get("midblock_forward_mse"),
                "midblock_forward_rel_mse": agg.get("midblock_forward_rel_mse"),
                "midblock_forward_rel_l2": agg.get("midblock_forward_rel_l2"),
                "midblock_forward_cosine": agg.get("midblock_forward_cosine"),
                "midblock_forward_max_abs": agg.get("midblock_forward_max_abs"),
                "hps": agg.get("hps"),
                "hps_fp": agg.get("hps_fp"),
                "n": agg.get("n"),
                "quant_build_s": timing.get("quant_build_s"),
                "eval_s": timing.get("eval_s"),
                "total_s": timing.get("total_s"),
            })
    os.replace(tmp_csv, vqdm_results_csv_path)


# %% cell 41
# === VQDM bpp sweep run ===

if "vqdm_xtx_stats" not in globals():
    raise RuntimeError("Run the VQDM setup cell first to collect vqdm_xtx_stats")
if "vqdm_fp_image_cache" not in globals():
    raise RuntimeError("Run the metric/cache cell first to precompute vqdm_fp_image_cache")

vqdm_sweep_results = []
vqdm_total_q_images = (
    len(vqdm_scope_specs)
    * len(vqdm_active_profiles)
    * len(vqdm_eval_prompts)
    * len(vqdm_eval_seeds_active)
)

with tqdm(total=vqdm_total_q_images, desc="VQDM bpp sweep images", unit="image") as image_pbar:
    for scope in vqdm_scope_specs:
        for profile in vqdm_active_profiles:
            profile_key = vqdm_profile_key(profile)
            profile_label = vqdm_profile_label(profile)
            label = f"VQDM {scope['label']} | {profile_label}"
            cfg = make_vqdm_aq_cfg(profile)

            t_variant0 = time.perf_counter()
            t_quant0 = time.perf_counter()
            pipe_vqdm_q, mid_block_q = build_vqdm_midblock_pipe_from_cfg(
                profile,
                cfg,
                label,
                target_layer_names=scope["layer_names"],
                include_conv=scope["include_conv"],
                xtx_stats=vqdm_xtx_stats,
            )
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t_quant = time.perf_counter() - t_quant0
            layer_diagnostics = collect_vqdm_layer_diagnostics(mid_block_q)
            code_usage_summary = summarize_vqdm_code_usage_from_diagnostics(layer_diagnostics)

            if vqdm_run_sanity_check and pipe_vqdm_q is not None:
                pipe_fp_device = vqdm_eval_device if vqdm_keep_eval_pipes_on_device else "cpu"
                pipe_fp_dtype = vqdm_final_dtype if pipe_fp_device == vqdm_eval_device else torch.float32
                pipe_fp = force_pipeline_device_dtype(copy.deepcopy(pipe), pipe_fp_device, pipe_fp_dtype)
                _ = sanity_check_vqdm_pipe(
                    pipe_fp=pipe_fp,
                    pipe_q=pipe_vqdm_q,
                    prompt=vqdm_eval_prompts[0],
                    seed=vqdm_eval_seeds_active[0],
                    num_inference_steps=vqdm_eval_num_inference_steps,
                    guidance_scale=vqdm_eval_guidance_scale,
                    device=vqdm_eval_device,
                    show_images=True,
                )
                del pipe_fp
            elif vqdm_run_sanity_check:
                print("VQDM sanity check skipped because in-place mid_block eval is enabled.")

            t_eval0 = time.perf_counter()
            if pipe_vqdm_q is None:
                agg, rows = eval_with_temporary_vqdm_midblock(
                    pipe_obj=pipe,
                    mid_block_q=mid_block_q,
                    prompts=vqdm_eval_prompts,
                    seeds=vqdm_eval_seeds_active,
                    fp_cache=vqdm_fp_image_cache,
                    image_pbar=image_pbar,
                    eval_label=label,
                )
            else:
                agg, rows = eval_vqdm_prompts_against_fp(
                    pipe,
                    pipe_vqdm_q,
                    vqdm_eval_prompts,
                    vqdm_eval_seeds_active,
                    fp_cache=vqdm_fp_image_cache,
                    image_pbar=image_pbar,
                    eval_label=label,
                )
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t_eval = time.perf_counter() - t_eval0
            t_total = time.perf_counter() - t_variant0
            log_event(f"Variant timing | {label} | quant_build={t_quant:.1f}s, eval={t_eval:.1f}s, total={t_total:.1f}s")
            score = agg["clip_ii"] - agg["lpips"]
            compression = scope["compression_by_profile"][profile_key]

            result = {
                "label": label,
                "scope_key": scope["key"],
                "scope_label": scope["label"],
                "target_mode": scope["target_mode"],
                "include_conv": bool(scope["include_conv"]),
                "target_layer_count": int(scope["target_layer_count"]),
                "target_linear_count": int(scope["target_linear_count"]),
                "target_conv_count": int(scope["target_conv_count"]),
                "target_layer_names": list(scope["layer_names"]),
                "profile_key": profile_key,
                "profile_label": profile_label,
                "reference_bpp": float(profile.get("reference_bpp", float("nan"))),
                "num_codebooks": int(profile["num_codebooks"]),
                "nbits_per_codebook": int(profile["nbits_per_codebook"]),
                "codebook_size": int(2 ** int(profile["nbits_per_codebook"])),
                "compression": compression,
                "score": score,
                "agg": agg,
                "rows": rows,
                "layer_diagnostics": layer_diagnostics,
                "code_usage_summary": code_usage_summary,
                "timing": {
                    "quant_build_s": float(t_quant),
                    "eval_s": float(t_eval),
                    "total_s": float(t_total),
                },
            }
            vqdm_sweep_results.append(result)
            save_vqdm_sweep_outputs(vqdm_sweep_results)

            print()
            print("#" * 110)
            print(
                f"VQDM RESULT | {label} | scope={scope['key']} | "
                f"estimated_bpp={compression['estimated_bpp']:.4f} | "
                f"score={score:.6f} | agg={agg}"
            )
            print(f"Saved partial VQDM results: {vqdm_results_json_path} and {vqdm_results_csv_path}")

            if pipe_vqdm_q is not None and not vqdm_keep_eval_pipes_on_device:
                move_pipeline_to_cpu_safely(pipe_vqdm_q)
            del pipe_vqdm_q, mid_block_q
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            log_cuda_memory(f"after {label}", vqdm_primary_device)

vqdm_sweep_results_sorted = sorted(
    vqdm_sweep_results,
    key=lambda r: (r["scope_key"], -float(r["reference_bpp"])),
)
vqdm_sweep_results_by_score = sorted(vqdm_sweep_results, key=lambda r: r["score"], reverse=True)

print()
print("#" * 110)
print("VQDM FINAL SUMMARY BY SCOPE/BPP")
for r in vqdm_sweep_results_sorted:
    agg = r["agg"]
    print(
        f"{r['scope_key']:32s} | ref_bpp~{r['reference_bpp']:.2f} | "
        f"M={r['num_codebooks']} | bits={r['nbits_per_codebook']} | K={r['codebook_size']} | "
        f"est_bpp={r['compression']['estimated_bpp']:.4f} | score={r['score']:.6f} | LPIPS={agg['lpips']:.4f} | "
        f"CLIP i-i={agg['clip_ii']:.4f} | CLIP t-i={agg['clip_ti']:.4f} | "
        f"HPS={agg['hps']:.4f} | HPS FP={agg['hps_fp']:.4f} | n={agg['n']}"
    )

print()
print("#" * 110)
print("VQDM FINAL TOP BY SCORE")
for idx, r in enumerate(vqdm_sweep_results_by_score[:20], 1):
    agg = r["agg"]
    print(
        f"{idx:2d}. {r['scope_key']} | ref_bpp~{r['reference_bpp']:.2f} | "
        f"est_bpp={r['compression']['estimated_bpp']:.4f} | score={r['score']:.6f} | "
        f"LPIPS={agg['lpips']:.4f} | CLIP i-i={agg['clip_ii']:.4f}"
    )

save_vqdm_sweep_outputs(vqdm_sweep_results)
print(f"VQDM JSON: {vqdm_results_json_path}")
print(f"VQDM CSV : {vqdm_results_csv_path}")

