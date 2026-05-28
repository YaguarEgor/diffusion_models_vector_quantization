# -*- coding: utf-8 -*-
# Unified RVQ runner: one-file SD1.5 RVQ/VQDM-like experiment with adaptive GPU defaults.

# %% [markdown] cell 1
# <h1>Все импорты и конфиги

# %% cell 2
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
LOGGER = logging.getLogger("quantize-vector")


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


def log_cuda_memory(label: str = ""):
    try:
        import torch

        if not torch.cuda.is_available():
            return
        free, total = torch.cuda.mem_get_info()
        allocated = torch.cuda.memory_allocated()
        reserved = torch.cuda.memory_reserved()
        suffix = f" | {label}" if label else ""
        LOGGER.info(
            "CUDA memory%s: allocated=%.2f GiB, reserved=%.2f GiB, free=%.2f GiB, total=%.2f GiB",
            suffix,
            allocated / 1024**3,
            reserved / 1024**3,
            free / 1024**3,
            total / 1024**3,
        )
    except Exception as exc:
        LOGGER.info("CUDA memory logging skipped: %s: %s", type(exc).__name__, exc)


def _finish_logging():
    elapsed = _format_seconds(time.time() - SCRIPT_STARTED_AT)
    LOGGER.info("Run finished in %s", elapsed)


def _log_uncaught_exception(exc_type, exc, tb):
    LOGGER.critical("Uncaught exception", exc_info=(exc_type, exc, tb))


atexit.register(_finish_logging)
sys.excepthook = _log_uncaught_exception

log_section("Script start")

IN_KAGGLE = "KAGGLE_KERNEL_RUN_TYPE" in os.environ
print("Kaggle detected:" if IN_KAGGLE else "Local/other runtime detected:", IN_KAGGLE)
if _REMOVED_USER_SITE_PATHS:
    print("Ignored user-site package paths:", _REMOVED_USER_SITE_PATHS)

_REQUIRED_PACKAGES = [
    ("numpy", "numpy"),
    ("pandas", "pandas"),
    ("matplotlib", "matplotlib"),
    ("tqdm", "tqdm"),
    ("PIL", "Pillow"),
    ("diffusers", "diffusers"),
    ("transformers", "transformers"),
    ("accelerate", "accelerate"),
    ("safetensors", "safetensors"),
    ("open_clip", "open_clip_torch"),
    ("lpips", "lpips"),
    ("hpsv2", "hpsv2"),
    ("huggingface_hub", "huggingface_hub"),
]

missing = [pkg for module_name, pkg in _REQUIRED_PACKAGES if importlib.util.find_spec(module_name) is None]
if missing:
    print("Installing missing packages:", missing)
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *missing])
    except Exception:
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


HPSV2_BPE_READY = ensure_hpsv2_bpe()
log_event(f"HPSv2 BPE ready: {HPSV2_BPE_READY}")

# %% cell 3
# === Импорты: стандартная библиотека ===
import os, csv, math, copy, json, hashlib, random
from pathlib import Path
from typing import Dict, Tuple, List, Optional, Any, Sequence
from dataclasses import dataclass

# === "Научные" библиотеки ===
import numpy as np

# === PyTorch: тензоры, модули и слои ===
import torch
import torch.nn as nn
import torch.nn.functional as F

# === Torchvision: датасеты и преобразования ===
import torchvision.transforms as T

# === Визуализация и прогресс-бары ===
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import time

# === Метрики ===
import open_clip
import lpips
import hpsv2

# === Stable Diffusion и Hugging Face ===
from diffusers import StableDiffusionPipeline, DDIMScheduler
from huggingface_hub import login

hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
if hf_token:
    login(token=hf_token)
else:
    print("HF_TOKEN is not set; using cached Hugging Face auth or public model access.")

# === Игнорировать предупреждения (бесят) ===
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", module="pydantic")


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
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


def _cuda_device_totals_gib_safe() -> List[float]:
    if not torch.cuda.is_available():
        return []
    totals = []
    for idx in range(torch.cuda.device_count()):
        try:
            totals.append(float(torch.cuda.get_device_properties(idx).total_memory) / 1024**3)
        except Exception:
            totals.append(0.0)
    return totals


def _auto_device_list_env(max_devices: int = 4) -> Optional[str]:
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        return None
    n = min(int(max_devices), torch.cuda.device_count())
    return ",".join(str(i) for i in range(n))


_AUTO_CUDA_TOTALS_GIB = _cuda_device_totals_gib_safe()
_AUTO_CUDA_COUNT = len(_AUTO_CUDA_TOTALS_GIB)
_AUTO_MAX_CUDA_GIB = max(_AUTO_CUDA_TOTALS_GIB) if _AUTO_CUDA_TOTALS_GIB else 0.0
_AUTO_MULTI_GPU = _AUTO_CUDA_COUNT >= 2
_AUTO_SMALL_MULTI_GPU_BOX = _AUTO_MULTI_GPU and _AUTO_MAX_CUDA_GIB < 48.0
_AUTO_HUGE_GPU = _AUTO_MAX_CUDA_GIB >= 70.0
_AUTO_DEVICE_LIST_ENV = _auto_device_list_env(
    max_devices=_env_int("RVQ_AUTO_MAX_DEVICES", max(1, _AUTO_CUDA_COUNT))
)


RVQ_HW_PROFILE = os.environ.get("RVQ_HW_PROFILE", "auto").strip().lower()
if RVQ_HW_PROFILE not in {"auto", "h100", "v100x4", "balanced"}:
    raise ValueError("RVQ_HW_PROFILE must be one of: auto, h100, v100x4, balanced")

if RVQ_HW_PROFILE == "h100":
    _AUTO_SMALL_MULTI_GPU_BOX = False
    _AUTO_HUGE_GPU = True
    _AUTO_MAX_CUDA_GIB = max(_AUTO_MAX_CUDA_GIB, 80.0)
elif RVQ_HW_PROFILE == "v100x4":
    _AUTO_SMALL_MULTI_GPU_BOX = True
    _AUTO_HUGE_GPU = False
    _AUTO_DEVICE_LIST_ENV = _AUTO_DEVICE_LIST_ENV or "0,1,2,3"
elif RVQ_HW_PROFILE == "balanced":
    _AUTO_SMALL_MULTI_GPU_BOX = False
    _AUTO_HUGE_GPU = False


RVQ_GPU_AUTOTUNE = _env_bool("RVQ_GPU_AUTOTUNE", True)
RVQ_BATCH_BINSEARCH = _env_bool("RVQ_BATCH_BINSEARCH", True)
RVQ_XTX_BINSEARCH = _env_bool("RVQ_XTX_BINSEARCH", True)
RVQ_XTX_PROBE_STEPS = _env_int("RVQ_XTX_PROBE_STEPS", 2)
RVQ_ALLOW_TF32 = _env_bool("RVQ_ALLOW_TF32", True)
RVQ_GPU_TARGET_MEM_FRACTION = _env_float("RVQ_GPU_TARGET_MEM_FRACTION", 0.58 if _AUTO_SMALL_MULTI_GPU_BOX else 0.70)
RVQ_ASSIGN_MIN_BATCH = _env_int("RVQ_ASSIGN_MIN_BATCH", 4096 if _AUTO_SMALL_MULTI_GPU_BOX else 8192)
RVQ_ASSIGN_MAX_BATCH = _env_int("RVQ_ASSIGN_MAX_BATCH", 524_288 if _AUTO_SMALL_MULTI_GPU_BOX else 2_097_152)
RVQ_USE_MULTI_GPU_ASSIGN = _env_bool("RVQ_USE_MULTI_GPU_ASSIGN", True)
RVQ_MULTI_GPU_MIN_N = _env_int("RVQ_MULTI_GPU_MIN_N", 65_536 if _AUTO_SMALL_MULTI_GPU_BOX else 262_144)
RVQ_MULTI_GPU_DEVICES_ENV = os.environ.get("RVQ_MULTI_GPU_DEVICES") or os.environ.get("RVQ_DEVICES") or _AUTO_DEVICE_LIST_ENV
RVQ_USE_MULTI_GPU_XTX = _env_bool("RVQ_USE_MULTI_GPU_XTX", True)
RVQ_XTX_DEVICES_ENV = os.environ.get("RVQ_XTX_DEVICES") or RVQ_MULTI_GPU_DEVICES_ENV
RVQ_XTX_MAX_BATCH = _env_int("RVQ_XTX_MAX_BATCH", 8 if _AUTO_SMALL_MULTI_GPU_BOX else 0)  # 0 = all calibration prompts
RVQ_CONV_XTX_MAX_BATCH = _env_int("RVQ_CONV_XTX_MAX_BATCH", 2 if _AUTO_SMALL_MULTI_GPU_BOX else 4)
RVQ_XTX_ESTIMATED_GIB_PER_PROMPT = _env_float("RVQ_XTX_ESTIMATED_GIB_PER_PROMPT", 1.75 if _AUTO_SMALL_MULTI_GPU_BOX else 1.25)
RVQ_HIGH_VRAM_THRESHOLD_GIB = _env_float("RVQ_HIGH_VRAM_THRESHOLD_GIB", 24.0 if _AUTO_MAX_CUDA_GIB < 48.0 else 40.0)
# Inference/evaluation acceleration knobs.
# RVQ_GEN_BATCH > 1 batches Stable Diffusion generations during sweep/eval.
# RVQ_COMPUTE_HPS is disabled by default because sweep ranking uses CLIP image-image - LPIPS,
# while HPS is CPU-heavy and does not affect model selection in this script.
RVQ_GEN_BATCH_ENV = _env_int("RVQ_GEN_BATCH", 0)  # 0 = auto
RVQ_GEN_BATCH_MAX = _env_int("RVQ_GEN_BATCH_MAX", 8 if _AUTO_SMALL_MULTI_GPU_BOX else 0)  # 0 = auto cap
RVQ_COMPUTE_HPS_ENV = _env_bool("RVQ_COMPUTE_HPS", False)
RVQ_STORE_COMPRESSED_ENV = os.environ.get("RVQ_STORE_COMPRESSED")
RVQ_MATERIALIZE_COMPRESSED_FOR_EVAL = _env_bool("RVQ_MATERIALIZE_COMPRESSED_FOR_EVAL", True)
RVQ_METRIC_DEVICE_ENV = os.environ.get("RVQ_METRIC_DEVICE")
RVQ_LOG_XTX_BLOCK_ENERGY = _env_bool("RVQ_LOG_XTX_BLOCK_ENERGY", True)
RVQ_FULL_XTX_RECODE_PASSES = _env_int("RVQ_FULL_XTX_RECODE_PASSES", 1)
RVQ_FULL_XTX_RECODE_ROW_BATCH = _env_int("RVQ_FULL_XTX_RECODE_ROW_BATCH", 0)  # 0 = all output rows
RVQ_FULL_XTX_RECODE_MAX_COLS = _env_int("RVQ_FULL_XTX_RECODE_MAX_COLS", 0)  # 0 = uncapped
RVQ_HADAMARD_MODE_ENV = os.environ.get("RVQ_HADAMARD_MODE", "auto").strip().lower()
RVQ_STAGE_K_SWEEP = _env_bool("RVQ_STAGE_K_SWEEP", False)
RVQ_FUNNEL_SWEEP = _env_bool("RVQ_FUNNEL_SWEEP", False)
RVQ_FUNNEL_K_VALUES_ENV = os.environ.get("RVQ_FUNNEL_K_VALUES", "128,256,512")
RVQ_FUNNEL_BLOCK_TOPK = _env_int("RVQ_FUNNEL_BLOCK_TOPK", 2)  # 0 = use all Hadamard blocks
RVQ_FUNNEL_FINAL_TOPK = _env_int("RVQ_FUNNEL_FINAL_TOPK", 3)  # per bpp and per H/noH track
RVQ_FUNNEL_INCLUDE_CONV_STAGE = _env_bool("RVQ_FUNNEL_INCLUDE_CONV_STAGE", True)
RVQ_FUNNEL_CONV_TOPK = _env_int("RVQ_FUNNEL_CONV_TOPK", 1)  # per bpp and per H/noH track
RVQ_FUNNEL_RESULTS_JSON = os.environ.get("RVQ_FUNNEL_RESULTS_JSON", "rvq_funnel_results.json")
RVQ_FIXED_SWEEP_OUTPUT_DIR = os.environ.get("RVQ_FIXED_SWEEP_OUTPUT_DIR", "").strip()
# Target selection for the RVQ sweep:
#   hardcoded_midblock    - quantize the fixed 10 VQDM-style Linear layers; add 6 convs with RVQ_INCLUDE_CONV=1 (16 layers)
#   all_midblock          - quantize all mid_block Linear layers and, with RVQ_INCLUDE_CONV=1, all Conv2d layers (18 layers)
#   all_midblock_linear   - quantize all VQDM-style Linear layers found in mid_block, excluding time_emb_proj
#   single_largest_linear - quantize only the largest Linear layer in mid_block
#   single_named_linear   - quantize only RVQ_SINGLE_TARGET_LAYER, with largest Linear fallback
RVQ_TARGET_MODE = os.environ.get("RVQ_TARGET_MODE", "all_midblock").strip().lower()
RVQ_SINGLE_TARGET_LAYER_ENV = os.environ.get("RVQ_SINGLE_TARGET_LAYER")
RVQ_INCLUDE_CONV_ENV = _env_bool("RVQ_INCLUDE_CONV", True)


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


def _cuda_mem_info(device: Optional[torch.device | str | int] = None):
    if not torch.cuda.is_available():
        return None
    dev = None
    if device is not None:
        dev = torch.device(f"cuda:{device}") if isinstance(device, int) else torch.device(device)
        if dev.type != "cuda":
            return None
        dev = torch.device(f"cuda:{torch.cuda.current_device()}" if dev.index is None else f"cuda:{dev.index}")
    free, total = torch.cuda.mem_get_info(dev)
    return {
        "free": int(free),
        "total": int(total),
        "free_gib": free / 1024**3,
        "total_gib": total / 1024**3,
        "device": str(dev) if dev is not None else f"cuda:{torch.cuda.current_device()}",
    }


def _cuda_total_gib(default: float = 0.0, device: Optional[torch.device | str | int] = None) -> float:
    info = _cuda_mem_info(device)
    return default if info is None else float(info["total_gib"])


def _is_big_gpu(min_total_gib: float = 40.0, device: Optional[torch.device | str | int] = None) -> bool:
    return torch.cuda.is_available() and _cuda_total_gib(device=device) >= min_total_gib


def _floor_to_multiple(x: int, multiple: int) -> int:
    x = max(int(x), 1)
    multiple = max(int(multiple), 1)
    return max(multiple, (x // multiple) * multiple)


def _format_batch(batch: int) -> str:
    return f"{int(batch):,}".replace(",", "_")


_RVQ_ASSIGN_BATCH_CACHE: dict[tuple, int] = {}
_RVQ_XTX_BATCH_CACHE: dict[tuple, int] = {}
_RVQ_MULTIGPU_LOG_CACHE: set[tuple] = set()

if torch.cuda.is_available():
    torch.cuda.empty_cache()
    try:
        torch.cuda.ipc_collect()
    except Exception:
        pass

# %% cell 4
# === Load Stable Diffusion ===

torch.backends.cudnn.allow_tf32 = bool(RVQ_ALLOW_TF32)
torch.backends.cuda.matmul.allow_tf32 = bool(RVQ_ALLOW_TF32)

runtime_device = "cuda" if torch.cuda.is_available() else "cpu"
runtime_dtype = torch.float16 if runtime_device == "cuda" else torch.float32
if RVQ_METRIC_DEVICE_ENV:
    metric_device = RVQ_METRIC_DEVICE_ENV
elif runtime_device == "cuda" and torch.cuda.device_count() > 1:
    metric_device = f"cuda:{min(torch.cuda.device_count() - 1, 3)}"
else:
    metric_device = runtime_device if (runtime_device == "cuda" and (_is_big_gpu(RVQ_HIGH_VRAM_THRESHOLD_GIB) or not globals().get("IN_KAGGLE", False))) else "cpu"

if runtime_device == "cuda":
    gpu_name = torch.cuda.get_device_name(0)
    gpu_capability = torch.cuda.get_device_capability(0)
    print(f"cuda device: {gpu_name}, capability=sm_{gpu_capability[0]}{gpu_capability[1]}")
    if gpu_capability[0] < 7:
        print(
            "WARNING: this GPU has compute capability < 7.0. "
            "Recent Kaggle/PyTorch CUDA builds may fail with 'no kernel image is available'. "
            "Use a T4/A100-class runtime for this notebook."
        )

log_section("Load Stable Diffusion and metric models")
log_event(
    f"runtime_device={runtime_device}, runtime_dtype={runtime_dtype}, metric_device={metric_device}, "
    f"hw_profile={RVQ_HW_PROFILE}, cuda_totals_gib={[round(x, 1) for x in _AUTO_CUDA_TOTALS_GIB]}, "
    f"gpu_autotune={RVQ_GPU_AUTOTUNE}, batch_binsearch={RVQ_BATCH_BINSEARCH}, "
    f"xtx_binsearch={RVQ_XTX_BINSEARCH}, allow_tf32={RVQ_ALLOW_TF32}, "
    f"target_mem_fraction={RVQ_GPU_TARGET_MEM_FRACTION}, "
    f"multi_gpu_assign={RVQ_USE_MULTI_GPU_ASSIGN}, multi_gpu_xtx={RVQ_USE_MULTI_GPU_XTX}, "
    f"multi_gpu_devices={RVQ_MULTI_GPU_DEVICES_ENV or 'auto'}, xtx_devices={RVQ_XTX_DEVICES_ENV or 'auto'}"
)
log_cuda_memory("before model load")

pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=runtime_dtype,
    safety_checker=None,
).to(runtime_device)
pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
pipe.set_progress_bar_config(disable=False)
log_cuda_memory("after Stable Diffusion load")

clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
    'ViT-H-14',
    pretrained='laion2b_s32b_b79k'
)
clip_model = clip_model.to(metric_device).eval()
clip_tokenizer = open_clip.get_tokenizer('ViT-H-14')
lpips_model = lpips.LPIPS(net='alex').to(metric_device).eval()
log_cuda_memory("after metric model load")

print("runtime_device:", runtime_device)
print("runtime_dtype:", runtime_dtype)
print("metric_device:", metric_device)

# %% cell 5
pipe.unet

# %% cell 6
# === Конфиг векторной квантизации ===

@dataclass 
class VectorizeMeta: 
    orig_shape: Tuple[int, ...]
    num_el: int
    d: int
    pad: int
    prefix_shape: Tuple[int, ...]   # shape после flatten всех осей кроме последней
    last_dim: int                   # размер последней оси после flatten
    n_chunks: int                   # сколько кусков по последней оси

@dataclass
class VQConfig:
    bits: int = 4
    d: int = 16
    K_override: Optional[int] = None
    stage_Ks: Optional[Tuple[int, ...]] = None

    max_iter: int = 50
    tol: float = 1e-5
    batch_assign: int = 65536
    seed: int = 5

    on_cpu: bool = False

    use_hadamard: bool = True
    hadamard_block: int = 16
    hadamard_with_signs: bool = True   # D * H (Rademacher signs)
    hadamard_seed: int = 123

    rvq_M: int = 2                 
    rvq_first_stage_n_init: int = 4 # k-means restarts for every RVQ stage (1 disables multistart)

    use_activation_aware: bool = False
    xtx_damping: float = 1e-6
    xtx_normalize_blocks: bool = False
    log_xtx_block_energy: bool = True
    full_xtx_recode_passes: int = 1
    full_xtx_recode_row_batch: int = 0
    full_xtx_recode_max_cols: int = 0
    group_channels: bool = True       # для Conv2d: VQDM-style order [kh, kw, cin]

    show_inner_progress: bool = True
    progress_leave: bool = False
    free_xtx_after_layer: bool = False
    store_compressed: bool = True  # store RVQ codes/codebooks instead of dense quantized weights

    def K(self) -> int:
        return int(self.K_override) if self.K_override is not None else (1 << int(self.bits))


def _cfg_stage_ks(cfg: VQConfig) -> Tuple[int, ...]:
    if cfg.stage_Ks is not None:
        values = tuple(int(k) for k in cfg.stage_Ks)
        if not values:
            raise ValueError("stage_Ks must not be empty")
        if any(k < 2 for k in values):
            raise ValueError(f"stage_Ks entries must be >= 2, got {values}")
        return values
    return tuple(int(cfg.K()) for _ in range(max(int(cfg.rvq_M), 1)))


def _stage_bits_for_ks(stage_ks: Sequence[int]) -> Tuple[int, ...]:
    return tuple(int(max(1, int(k)) - 1).bit_length() if int(k) > 1 else 1 for k in stage_ks)


def _cfg_stage_bits(cfg: VQConfig) -> Tuple[int, ...]:
    return _stage_bits_for_ks(_cfg_stage_ks(cfg))


def _cfg_index_bits_per_vector(cfg: VQConfig) -> int:
    return int(sum(_cfg_stage_bits(cfg)))


def _cfg_codebook_entries(cfg: VQConfig) -> int:
    return int(sum(_cfg_stage_ks(cfg)))


def _format_stage_ks(stage_ks: Sequence[int]) -> str:
    return "[" + ",".join(str(int(k)) for k in stage_ks) + "]"


def count_quantizable_layers(module: nn.Module):
    n = 0
    for m in module.modules():
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            n += 1
    return n


VQC = VQConfig()

# %% cell 7
# === Подсчет того, как сильно квантуется модель ===

def _weight_stats(layer: nn.Module, d: int) -> Dict[str, int]:
    """
    Считает статистику для weight любого слоя, который будем квантизовать
    построчно после flatten по всем осям, кроме первой.

    Linear:
        weight shape = (out_features, in_features)

    Conv2d:
        weight shape = (out_channels, in_channels, kh, kw)
        -> flatten(1) => (out_channels, in_channels * kh * kw)
    """
    W = layer.weight.detach()
    out_dim = W.shape[0]
    flat_in_dim = W[0].numel()

    n_chunks = math.ceil(flat_in_dim / d)
    padded_last_dim = n_chunks * d
    pad_per_row = padded_last_dim - flat_in_dim

    orig_params = W.numel()
    padded_params = out_dim * padded_last_dim
    total_pad = out_dim * pad_per_row
    n_vectors = out_dim * n_chunks

    return {
        "orig_params": orig_params,
        "padded_params": padded_params,
        "pad_params": total_pad,
        "n_vectors": n_vectors,
        "out_dim": out_dim,
        "flat_in_dim": flat_in_dim,
        "n_chunks": n_chunks,
    }


def compute_rvq_compression(
    module: nn.Module,
    cfg: VQConfig,
    fp_bits: int = 16,
    codebook_bits: int = 16,
    quantize_bias: bool = False,
    bias_bits: Optional[int] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Считает compression stats для RVQ, если квантуются nn.Linear и nn.Conv2d слои.
    """
    d = int(cfg.d)
    stage_ks = _cfg_stage_ks(cfg)
    stage_bits = _cfg_stage_bits(cfg)
    M = len(stage_ks)
    K = max(stage_ks)
    bits = max(stage_bits)
    index_bits_per_vector = sum(stage_bits)
    codebook_entries = sum(stage_ks)

    if bias_bits is None:
        bias_bits = fp_bits

    total_model_params = 0
    total_model_fp_bits = 0

    total_quant_weight_params = 0
    total_quant_weight_fp_bits = 0

    total_bias_params = 0
    total_bias_bits = 0

    total_index_bits = 0
    total_codebook_bits = 0
    total_quant_bits = 0
    total_pad_params = 0
    total_vectors = 0

    per_layer = []

    for name, m in module.named_modules():
        for p in m.parameters(recurse=False):
            total_model_params += p.numel()
            total_model_fp_bits += p.numel() * fp_bits

        if not isinstance(m, (nn.Linear, nn.Conv2d)):
            continue

        stats = _weight_stats(m, d=d)

        weight_params = stats["orig_params"]
        n_vectors = stats["n_vectors"]
        pad_params = stats["pad_params"]

        index_bits = n_vectors * index_bits_per_vector
        cb_bits = codebook_entries * d * codebook_bits

        layer_bias_params = m.bias.numel() if m.bias is not None else 0
        layer_bias_bits = layer_bias_params * bias_bits

        layer_quant_bits = index_bits + cb_bits + layer_bias_bits

        total_quant_weight_params += weight_params
        total_quant_weight_fp_bits += weight_params * fp_bits

        total_bias_params += layer_bias_params
        total_bias_bits += layer_bias_bits

        total_index_bits += index_bits
        total_codebook_bits += cb_bits
        total_quant_bits += layer_quant_bits

        total_pad_params += pad_params
        total_vectors += n_vectors

        per_layer.append({
            "name": name,
            "layer_type": type(m).__name__,
            "weight_shape": tuple(m.weight.shape),
            "weight_params": weight_params,
            "bias_params": layer_bias_params,
            "n_vectors": n_vectors,
            "pad_params": pad_params,
            "index_bits": index_bits,
            "codebook_bits": cb_bits,
            "bias_bits": layer_bias_bits,
            "quant_bits_total": layer_quant_bits,
            "fp_bits_weight_only": weight_params * fp_bits,
            "compression_ratio_weight_only": (weight_params * fp_bits) / (index_bits + cb_bits) if (index_bits + cb_bits) > 0 else float("inf"),
            "bpp_weight_only": (index_bits + cb_bits) / weight_params if weight_params > 0 else 0.0,
        })

    quant_weight_bpp = (
        (total_index_bits + total_codebook_bits) / total_quant_weight_params
        if total_quant_weight_params > 0 else 0.0
    )

    quant_weight_compression_ratio = (
        total_quant_weight_fp_bits / (total_index_bits + total_codebook_bits)
        if (total_index_bits + total_codebook_bits) > 0 else float("inf")
    )

    non_quantized_bits = total_model_fp_bits - total_quant_weight_fp_bits
    full_model_quant_bits = non_quantized_bits + total_quant_bits

    full_model_compression_ratio = (
        total_model_fp_bits / full_model_quant_bits
        if full_model_quant_bits > 0 else float("inf")
    )

    result = {
        "config": {
            "bits": bits,
            "K": K,
            "stage_Ks": list(stage_ks),
            "stage_bits": list(stage_bits),
            "d": d,
            "rvq_M": M,
            "fp_bits": fp_bits,
            "codebook_bits": codebook_bits,
            "quantize_bias": quantize_bias,
            "bias_bits": bias_bits,
        },
        "totals": {
            "total_model_params": total_model_params,
            "total_model_fp_bits": total_model_fp_bits,

            "total_quant_weight_params": total_quant_weight_params,
            "total_quant_weight_fp_bits": total_quant_weight_fp_bits,

            "total_bias_params_in_quant_layers": total_bias_params,
            "total_bias_bits_in_quant_layers": total_bias_bits,

            "total_vectors": total_vectors,
            "total_pad_params": total_pad_params,

            "total_index_bits": total_index_bits,
            "total_codebook_bits": total_codebook_bits,
            "total_quant_bits_including_bias": total_quant_bits,

            "quant_weight_bpp": quant_weight_bpp,
            "quant_weight_compression_ratio": quant_weight_compression_ratio,

            "full_model_quant_bits": full_model_quant_bits,
            "full_model_compression_ratio": full_model_compression_ratio,
        },
        "per_layer": per_layer,
    }

    if verbose:
        t = result["totals"]
        print("=== RVQ compression stats (Linear + Conv2d) ===")
        print(f"Quantized weight params         : {t['total_quant_weight_params']:,}")
        print(f"Bias params in quant layers     : {t['total_bias_params_in_quant_layers']:,}")
        print(f"Total RVQ vectors               : {t['total_vectors']:,}")
        print(f"Total padded params             : {t['total_pad_params']:,}")
        print()
        print(f"Index bits                      : {t['total_index_bits']:,}")
        print(f"Codebook bits                   : {t['total_codebook_bits']:,}")
        print(f"Quantized weight FP bits        : {t['total_quant_weight_fp_bits']:,}")
        print()
        print(f"Quantized weight bpp            : {t['quant_weight_bpp']:.6f}")
        print(f"Quantized weight compression    : {t['quant_weight_compression_ratio']:.4f}x")
        print()
        print(f"Full model FP bits              : {t['total_model_fp_bits']:,}")
        print(f"Full model quantized bits       : {t['full_model_quant_bits']:,}")
        print(f"Full model compression          : {t['full_model_compression_ratio']:.4f}x")

    return result

# %% [markdown] cell 8
# <h1>Разбиение матрицы весов на вектора для последующего квантования (и обратный сбор соответственно)</h1>

# %% cell 9
# === Функции для получения векторов для квантизации из весов с помощью "тупого" разрезания на одинаковые куски flattened вектора весов === 

@torch.no_grad()
def weight_to_vectors_lastdim(w: torch.Tensor, d: int):
    """
    Любой weight -> 2D матрица [rows, last_dim], затем режем по последней оси на куски длины d.

    Linear:
        [out, in] -> rows=out, last_dim=in

    Conv2d:
        [out, in, kh, kw] -> [out, in*kh*kw]
    """
    W = w.detach().float()
    assert W.dim() >= 2, "Ожидаю weight хотя бы размерности 2"

    orig_shape = tuple(W.shape)
    rows = W.shape[0]
    last_dim = W[0].numel()

    W2 = W.reshape(rows, last_dim)

    num_el = W2.numel()
    n_chunks = (last_dim + d - 1) // d
    last_dim_pad = n_chunks * d
    pad = last_dim_pad - last_dim

    if pad > 0:
        W2 = torch.nn.functional.pad(W2, (0, pad))

    X = W2.view(rows, n_chunks, d).reshape(-1, d)

    meta = VectorizeMeta(
        orig_shape=orig_shape,
        num_el=num_el,
        d=d,
        pad=pad,
        prefix_shape=(rows,),
        last_dim=last_dim,
        n_chunks=n_chunks,
    )
    return X, meta


@torch.no_grad()
def vectors_to_weight_lastdim(X: torch.Tensor, meta: VectorizeMeta, dtype=None):
    """
    X [rows*n_chunks, d] + meta -> weight исходной формы
    """
    d = meta.d
    assert X.dim() == 2 and X.shape[1] == d, "X должен быть [N, d]"

    rows = meta.prefix_shape[0]

    assert X.shape[0] == rows * meta.n_chunks, (
        f"X.shape[0]={X.shape[0]} не согласован с rows={rows} и n_chunks={meta.n_chunks}"
    )

    W2 = X.detach().float().view(rows, meta.n_chunks, d).reshape(rows, meta.n_chunks * d)

    if meta.pad > 0:
        W2 = W2[:, :meta.last_dim]

    W = W2.view(*meta.orig_shape)

    if dtype is not None:
        W = W.to(dtype)
    return W

# %% [markdown] cell 10
# <h1>Трансформация Адамара (ортогональная трансформация)</h1>

# %% cell 11
# === Трансформация Fast Walsh–Hadamard Transform (FWHT) с Rademacher signs ===

@torch.no_grad()
def fwht_(x: torch.Tensor) -> torch.Tensor:
    """
    Корректный FWHT по последней оси (n=2^m).
    Безопасно для in-place (использует clone() внутри баттерфляя).
    x: (..., n)
    """
    n = x.shape[-1]
    assert (n & (n - 1)) == 0, "FWHT requires last dim to be power of 2"

    orig_shape = x.shape
    y = x.contiguous().view(-1, n)  # [M, n]

    h = 1
    while h < n:
        y = y.view(-1, n // (2 * h), 2 * h)  # [M, blocks, 2h]

        a = y[..., :h].clone()
        b = y[..., h:2*h].clone()

        y[..., :h]     = a + b
        y[..., h:2*h]  = a - b

        y = y.view(-1, n)
        h *= 2

    x.copy_(y.view(orig_shape))
    return x

@torch.no_grad()
def rademacher_signs(n: int, device, seed: int, dtype=torch.float32) -> torch.Tensor:
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    s = torch.randint(0, 2, (n,), generator=g, device=device, dtype=torch.int8)
    s = (s * 2 - 1).to(dtype)
    return s

# %% cell 12
# === Домножение на матрицу Адамара и потом на обратную матрицу для "возвращения" к начальному состоянию ===

@torch.no_grad()
def _apply_hadamard_to_weight_rows(
    W: torch.Tensor,
    block: int,
    seed: int,
    with_signs: bool = True,
    inverse: bool = False,
):
    """
    Применяет блочный Hadamard к weight после flatten всех осей, кроме первой:

    Linear: [out_features, in_features]
    Conv2d: [out_channels, in_channels, kh, kw] -> [out_channels, in_channels * kh * kw]

    Это важно для Conv2d: Hadamard должен работать по полному входному вектору фильтра,
    а не только по последней оси ядра kw.
    """
    assert W.dim() >= 2, "Ожидаю weight хотя бы размерности 2"
    assert (block & (block - 1)) == 0, "block must be power of 2"

    orig_shape = tuple(W.shape)
    rows = W.shape[0]
    W_rows = W.reshape(rows, -1).contiguous().clone()

    I = W_rows.shape[1]
    I2 = (I // block) * block
    if I2 == 0:
        return W

    W_main = W_rows[:, :I2].contiguous().view(rows, I2 // block, block)

    if not inverse and with_signs:
        s = rademacher_signs(block, device=W.device, seed=seed, dtype=W.dtype)
        W_main.mul_(s)

    fwht_(W_main)
    W_main.mul_(1.0 / math.sqrt(block))

    if inverse and with_signs:
        s = rademacher_signs(block, device=W.device, seed=seed, dtype=W.dtype)
        W_main.mul_(s)

    W_rows[:, :I2] = W_main.view(rows, I2)
    W.copy_(W_rows.view(orig_shape))
    return W


@torch.no_grad()
def hadamard_right_blocks(W: torch.Tensor, block: int, seed: int, with_signs: bool = True):
    """
    W <- W * (D H) блочно по flatten-входу слоя.
    """
    return _apply_hadamard_to_weight_rows(
        W,
        block=block,
        seed=seed,
        with_signs=with_signs,
        inverse=False,
    )


@torch.no_grad()
def hadamard_right_blocks_inv(W: torch.Tensor, block: int, seed: int, with_signs: bool = True):
    """
    Инверсия для forward = D -> H -> /sqrt(block):
      inverse = H -> /sqrt(block) -> D
    """
    return _apply_hadamard_to_weight_rows(
        W,
        block=block,
        seed=seed,
        with_signs=with_signs,
        inverse=True,
    )

# %% [markdown] cell 13
# <h1>Лойд++ для обучения кодбука</h1>

# %% cell 14
# === Lloyd++ с batched assignment и умной инициализацией ===

def resolve_assignment_batch(
    *,
    X: torch.Tensor,
    K: int,
    d: int,
    weighted: bool,
    requested_batch: int | None,
    device: Optional[torch.device | str | int] = None,
) -> int:
    N = int(X.shape[0])
    requested_batch = int(requested_batch or 0)

    if requested_batch > 0 or not (RVQ_GPU_AUTOTUNE and X.is_cuda and torch.cuda.is_available()):
        return max(1, min(N, requested_batch if requested_batch > 0 else 8192))

    info = _cuda_mem_info(device if device is not None else X.device)
    if info is None:
        return max(1, min(N, 8192))

    dtype_bytes = max(torch.tensor([], dtype=X.dtype).element_size(), 2)
    if weighted:
        # Optimized weighted assignment avoids materializing diff [B, K, d].
        # Main temporaries are dist2/middle [B, K] plus small xH/H slices.
        bytes_per_row = dtype_bytes * (4 * int(K) + int(d) * int(d) + 4 * int(d))
    else:
        # Unweighted assignment mainly materializes dots/dist2 [B, K].
        bytes_per_row = dtype_bytes * (3 * int(K) + int(d))

    target_bytes = max(512 * 1024**2, int(info["free"] * RVQ_GPU_TARGET_MEM_FRACTION))
    raw_batch = max(RVQ_ASSIGN_MIN_BATCH, target_bytes // max(bytes_per_row, 1))
    batch = min(N, RVQ_ASSIGN_MAX_BATCH, _floor_to_multiple(raw_batch, RVQ_ASSIGN_MIN_BATCH))
    batch = max(1, int(batch))

    log_event(
        f"Auto assignment batch: {_format_batch(batch)} | weighted={weighted} | "
        f"N={_format_batch(N)}, K={K}, d={d}, device={info['device']}, free={info['free_gib']:.2f} GiB"
    )
    return batch


def _rvq_has_conv_targets(pipe, layer_names: Optional[Sequence[str]]) -> bool:
    if not layer_names:
        return False
    names = set(layer_names)
    root = getattr(pipe, "unet", pipe)
    return any(name in names and isinstance(module, nn.Conv2d) for name, module in root.named_modules())


def resolve_xtx_batch_size(
    num_prompts: int,
    requested_batch: int | None = None,
    *,
    has_conv_targets: bool = False,
    device: Optional[torch.device | str | int] = None,
) -> int:
    num_prompts = max(int(num_prompts), 1)
    requested_batch = int(requested_batch or 0)
    conv_cap = max(1, int(RVQ_CONV_XTX_MAX_BATCH))
    max_batch_for_layers = min(num_prompts, conv_cap) if has_conv_targets else num_prompts

    if requested_batch > 0 or not (RVQ_GPU_AUTOTUNE and torch.cuda.is_available()):
        batch = max(1, min(max_batch_for_layers, requested_batch if requested_batch > 0 else 1))
        if has_conv_targets and requested_batch > batch:
            log_event(f"Clamped requested XTX batch {requested_batch} -> {batch} for Conv2d targets")
        return batch

    info = _cuda_mem_info(device)
    if info is None:
        return 1

    estimated = int((info["free_gib"] * RVQ_GPU_TARGET_MEM_FRACTION) / max(RVQ_XTX_ESTIMATED_GIB_PER_PROMPT, 0.1))
    max_batch = num_prompts if RVQ_XTX_MAX_BATCH <= 0 else min(num_prompts, RVQ_XTX_MAX_BATCH)
    if has_conv_targets:
        max_batch = min(max_batch, conv_cap)
    batch = max(1, min(max_batch, estimated))

    log_event(
        f"Auto XTX batch: {batch} | prompts={num_prompts}, device={info['device']}, free={info['free_gib']:.2f} GiB, "
        f"estimated_gib_per_prompt={RVQ_XTX_ESTIMATED_GIB_PER_PROMPT:.2f}, conv_targets={has_conv_targets}"
    )
    return batch


def _rvq_xtx_devices() -> List[torch.device]:
    if not RVQ_USE_MULTI_GPU_XTX or not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        return [torch.device(runtime_device)]
    max_devices = _env_int("RVQ_AUTO_MAX_DEVICES", torch.cuda.device_count())
    max_devices = max(1, min(int(max_devices), torch.cuda.device_count()))
    requested = _parse_cuda_device_ids(RVQ_XTX_DEVICES_ENV)
    available = list(range(torch.cuda.device_count()))
    if requested is None:
        requested = available[:max_devices]
    ids = [idx for idx in requested if idx in available]
    if not ids:
        ids = [torch.cuda.current_device()]
    primary = torch.cuda.current_device()
    ids = [primary] + [idx for idx in ids if idx != primary]
    ids = list(dict.fromkeys(ids))[:max_devices]
    return [torch.device(f"cuda:{idx}") for idx in ids]


@torch.no_grad()
def _weighted_dist2_no_diff(
    Xb: torch.Tensor,
    C: torch.Tensor,
    H_blocks: torch.Tensor,
    block_ids_b: torch.Tensor,
    cHc_blocks: Optional[torch.Tensor] = None,
    include_xHx: bool = True,
) -> torch.Tensor:
    """
    Fast weighted squared distances without materializing diff [B, K, d].

    For every vector x_b and centroid c_k computes:
        (x_b - c_k)^T H_b (x_b - c_k)
      = x_b^T H_b x_b - 2 x_b^T H_b c_k + c_k^T H_b c_k.

    This is much more GPU-friendly for small d and large B,K because it keeps the
    largest temporary at [B, K] instead of [B, K, d].
    """
    H_blocks = H_blocks.to(device=Xb.device, dtype=Xb.dtype)
    C = C.to(dtype=Xb.dtype)
    block_ids_b = block_ids_b.to(device=Xb.device, dtype=torch.long)

    Hb = H_blocks.index_select(0, block_ids_b)             # [B, d, d]
    xH = torch.einsum("bd,bdh->bh", Xb, Hb)               # [B, d]
    middle = xH @ C.t().contiguous()                       # [B, K]

    if cHc_blocks is None:
        # [num_blocks, K], tiny in this experiment: n_chunks * K.
        cHc_blocks = torch.einsum("kd,bdh,kh->bk", C, H_blocks, C)
    else:
        cHc_blocks = cHc_blocks.to(device=Xb.device, dtype=Xb.dtype)
    cHc = cHc_blocks.index_select(0, block_ids_b)          # [B, K]

    dist2 = cHc
    dist2 = dist2.add(middle, alpha=-2.0)
    if include_xHx:
        xHx = (xH * Xb).sum(dim=1)                         # [B]
        dist2 = dist2.add(xHx[:, None])
    return dist2


@torch.no_grad()
def _probe_assignment_batch(
    X: torch.Tensor,
    C: torch.Tensor,
    batch: int,
    *,
    H_blocks: Optional[torch.Tensor] = None,
    block_ids: Optional[torch.Tensor] = None,
) -> bool:
    batch = max(1, min(int(batch), int(X.shape[0])))
    try:
        Xb = X[:batch]
        if H_blocks is None or block_ids is None:
            c2 = (C * C).sum(dim=1)
            scores = Xb @ C.t().contiguous()
            scores.add_(c2[None, :], alpha=-0.5)
            _ = scores.argmax(dim=1)
        else:
            cHc_blocks = torch.einsum(
                "kd,bdh,kh->bk",
                C.to(dtype=Xb.dtype),
                H_blocks.to(device=Xb.device, dtype=Xb.dtype),
                C.to(dtype=Xb.dtype),
            )
            dist2 = _weighted_dist2_no_diff(
                Xb,
                C,
                H_blocks=H_blocks,
                block_ids_b=block_ids[:batch],
                cHc_blocks=cHc_blocks,
                include_xHx=False,
            )
            _ = dist2.argmin(dim=1)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        return True
    except RuntimeError as exc:
        if _is_cuda_oom(exc):
            _clear_cuda_after_oom()
            return False
        raise
    finally:
        _clear_cuda_after_oom()


def _binary_search_batch(
    *,
    low: int,
    high: int,
    step: int,
    probe_fn,
    label: str,
) -> int:
    low = max(int(low), 1)
    high = max(int(high), low)
    step = max(int(step), 1)

    if high <= low:
        return high

    lo_units = max(1, (low + step - 1) // step)
    hi_units = max(lo_units, (high + step - 1) // step)
    best_units = 0

    while lo_units <= hi_units:
        mid_units = (lo_units + hi_units) // 2
        candidate = min(high, mid_units * step)
        ok = probe_fn(candidate)
        log_event(f"{label} binsearch probe batch={_format_batch(candidate)} -> {'ok' if ok else 'OOM'}")
        if ok:
            best_units = mid_units
            lo_units = mid_units + 1
        else:
            hi_units = mid_units - 1

    if best_units <= 0:
        return low if probe_fn(low) else 1
    return min(high, max(low, best_units * step))


def autotune_assignment_batch(
    *,
    X: torch.Tensor,
    C: torch.Tensor,
    requested_batch: int | None,
    H_blocks: Optional[torch.Tensor] = None,
    block_ids: Optional[torch.Tensor] = None,
) -> int:
    requested_batch = int(requested_batch or 0)
    N = int(X.shape[0])
    weighted = H_blocks is not None and block_ids is not None

    if requested_batch > 0 or not (RVQ_GPU_AUTOTUNE and RVQ_BATCH_BINSEARCH and X.is_cuda and torch.cuda.is_available()):
        return resolve_assignment_batch(
            X=X,
            K=int(C.shape[0]),
            d=int(X.shape[1]),
            weighted=weighted,
            requested_batch=requested_batch,
        )

    cache_key = (
        str(X.device),
        str(X.dtype),
        int(N),
        int(C.shape[0]),
        int(X.shape[1]),
        bool(weighted),
        int(RVQ_ASSIGN_MAX_BATCH),
        int(RVQ_ASSIGN_MIN_BATCH),
    )
    if cache_key in _RVQ_ASSIGN_BATCH_CACHE:
        batch = min(N, _RVQ_ASSIGN_BATCH_CACHE[cache_key])
        log_event(f"Reusing assignment batch from cache: {_format_batch(batch)} | weighted={weighted}")
        return batch

    high = min(N, int(RVQ_ASSIGN_MAX_BATCH))
    low = min(high, max(1, int(RVQ_ASSIGN_MIN_BATCH)))
    batch = _binary_search_batch(
        low=low,
        high=high,
        step=max(1, int(RVQ_ASSIGN_MIN_BATCH)),
        probe_fn=lambda b: _probe_assignment_batch(
            X,
            C,
            b,
            H_blocks=H_blocks if weighted else None,
            block_ids=block_ids if weighted else None,
        ),
        label=f"assignment weighted={weighted}",
    )
    _RVQ_ASSIGN_BATCH_CACHE[cache_key] = batch
    log_event(
        f"Auto assignment batch selected by binsearch: {_format_batch(batch)} | "
        f"weighted={weighted}, N={_format_batch(N)}, K={int(C.shape[0])}, d={int(X.shape[1])}"
    )
    return batch


def _parse_cuda_device_ids(value: Optional[str]) -> Optional[List[int]]:
    if not value:
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


def _rvq_assignment_devices(X: torch.Tensor) -> List[torch.device]:
    if (
        not RVQ_USE_MULTI_GPU_ASSIGN
        or not X.is_cuda
        or not torch.cuda.is_available()
        or torch.cuda.device_count() < 2
        or int(X.shape[0]) < int(RVQ_MULTI_GPU_MIN_N)
    ):
        return [X.device]

    requested = _parse_cuda_device_ids(RVQ_MULTI_GPU_DEVICES_ENV)
    available = list(range(torch.cuda.device_count()))
    if requested is None:
        requested = available
    requested = [idx for idx in requested if idx in available]
    if not requested:
        return [X.device]

    primary = X.device.index
    if primary is None:
        primary = torch.cuda.current_device()
    ids = [primary] + [idx for idx in requested if idx != primary]
    ids = list(dict.fromkeys(ids))
    if len(ids) < 2:
        return [X.device]
    return [torch.device(f"cuda:{idx}") for idx in ids]


def _even_slices(n: int, parts: int) -> List[slice]:
    parts = max(1, int(parts))
    shard = (int(n) + parts - 1) // parts
    return [
        slice(start, min(start + shard, int(n)))
        for start in range(0, int(n), shard)
    ]


def _device_vram_weight(device: torch.device) -> float:
    info = _cuda_mem_info(device)
    if info is None:
        return 1.0
    return max(float(info.get("free_gib", 0.0)), float(info.get("total_gib", 0.0)) * 0.25, 1.0)


def _weighted_slices(n: int, weights: Sequence[float]) -> List[slice]:
    n = max(int(n), 0)
    weights = [max(float(w), 1e-6) for w in weights]
    if n <= 0 or not weights:
        return []
    if len(weights) == 1:
        return [slice(0, n)]

    total = sum(weights)
    raw = [n * w / total for w in weights]
    counts = [int(math.floor(x)) for x in raw]
    remainder = n - sum(counts)
    order = sorted(range(len(weights)), key=lambda i: raw[i] - counts[i], reverse=True)
    for idx in order[:remainder]:
        counts[idx] += 1

    slices = []
    start = 0
    for count in counts:
        stop = min(n, start + count)
        if start < stop:
            slices.append(slice(start, stop))
        start = stop
    return slices


@torch.no_grad()
def _prepare_multigpu_assignment_context(
    X: torch.Tensor,
    *,
    H_blocks: Optional[torch.Tensor] = None,
    block_ids: Optional[torch.Tensor] = None,
) -> Optional[Dict[str, Any]]:
    devices = _rvq_assignment_devices(X)
    if len(devices) < 2:
        return None

    weights = [_device_vram_weight(dev) for dev in devices]
    slices = _weighted_slices(int(X.shape[0]), weights)
    devices = devices[:len(slices)]
    primary = X.device

    x_shards = []
    block_id_shards = []
    h_shards = []
    for dev, sl in zip(devices, slices):
        x_shards.append(X[sl].to(dev, non_blocking=True).contiguous())
        if H_blocks is not None and block_ids is not None:
            block_id_shards.append(block_ids[sl].to(dev, non_blocking=True).contiguous())
            h_shards.append(H_blocks.to(dev, dtype=X.dtype, non_blocking=True).contiguous())

    cache_key = (tuple(str(d) for d in devices), int(X.shape[0]), int(X.shape[1]), bool(H_blocks is not None))
    if cache_key not in _RVQ_MULTIGPU_LOG_CACHE:
        _RVQ_MULTIGPU_LOG_CACHE.add(cache_key)
        shard_sizes = ", ".join(f"{str(dev)}:{shard.shape[0]:,}" for dev, shard in zip(devices, x_shards))
        log_event(f"VQDM-style multi-GPU assignment enabled | shards: {shard_sizes}")

    return {
        "devices": devices,
        "slices": slices,
        "primary": primary,
        "x_shards": x_shards,
        "h_shards": h_shards if h_shards else None,
        "block_id_shards": block_id_shards if block_id_shards else None,
        "local_batch_cache": {},
    }


@torch.no_grad()
def _assign_multigpu_from_context(
    C: torch.Tensor,
    *,
    batch: int,
    context: Dict[str, Any],
    weighted: bool,
) -> torch.Tensor:
    from concurrent.futures import ThreadPoolExecutor

    devices: List[torch.device] = context["devices"]
    slices: List[slice] = context["slices"]
    x_shards: List[torch.Tensor] = context["x_shards"]
    primary: torch.device = context["primary"]
    h_shards = context.get("h_shards")
    block_id_shards = context.get("block_id_shards")
    local_batch_cache = context.setdefault("local_batch_cache", {})
    explicit_batch = int(globals().get("rvq_batch_assign", 0) or 0) > 0 or not RVQ_GPU_AUTOTUNE

    def worker(idx: int) -> torch.Tensor:
        dev = devices[idx]
        with torch.cuda.device(dev):
            Xd = x_shards[idx]
            Cd = C.to(dev, dtype=Xd.dtype, non_blocking=False).contiguous()
            if explicit_batch:
                local_batch = max(1, min(int(batch), int(Xd.shape[0])))
            else:
                cache_key = (
                    str(dev),
                    int(Xd.shape[0]),
                    int(Cd.shape[0]),
                    int(Cd.shape[1]),
                    str(Xd.dtype),
                    bool(weighted),
                )
                if cache_key not in local_batch_cache:
                    local_batch_cache[cache_key] = resolve_assignment_batch(
                        X=Xd,
                        K=int(Cd.shape[0]),
                        d=int(Cd.shape[1]),
                        weighted=bool(weighted),
                        requested_batch=0,
                        device=dev,
                    )
                local_batch = max(1, min(int(local_batch_cache[cache_key]), int(Xd.shape[0])))
            if weighted:
                labels_d = assign_to_centroids_batched_weighted(
                    Xd,
                    Cd,
                    H_blocks=h_shards[idx],
                    block_ids=block_id_shards[idx],
                    batch=local_batch,
                )
            else:
                labels_d = assign_to_centroids_batched(Xd, Cd, batch=local_batch)
            torch.cuda.synchronize(dev)
            return labels_d.to(primary, non_blocking=False)

    labels = torch.empty(sum(shard.shape[0] for shard in x_shards), device=primary, dtype=torch.long)
    with ThreadPoolExecutor(max_workers=len(devices)) as executor:
        futures = [executor.submit(worker, idx) for idx in range(len(devices))]
        for sl, fut in zip(slices, futures):
            labels[sl] = fut.result()
    if primary.type == "cuda":
        torch.cuda.synchronize(primary)
    return labels


@torch.no_grad() 
def init_kmeans_pp(X: torch.Tensor, K: int, seed: int = 5): 
    N, d = X.shape 
    g = torch.Generator(device=X.device) 
    g.manual_seed(seed) 
    idx0 = torch.randint(0, N, (1,), generator=g, device=X.device).item() 
    C = X[idx0:idx0+1].clone() 
    D2 = torch.sum((X - C[0]) ** 2, dim=1) 
    for _ in tqdm(range(K-1), desc="kmeans++ init", leave=False, unit="centroid"): 
        s = D2.sum() 
        if s.item() <= 0: 
            idx = torch.randint(0, N, (1,), generator=g, device=X.device).item() 
        else: 
            probs = D2 / s 
            idx = torch.multinomial(probs, 1, generator=g).item() 
        C = torch.cat([C, X[idx:idx+1]], dim=0) 
        d2_new = torch.sum((X - C[-1]) ** 2, dim=1) 
        D2 = torch.minimum(D2, d2_new) 
    return C 
    

@torch.no_grad() 
def assign_to_centroids_batched(X: torch.Tensor, C: torch.Tensor, batch: int = 65536): 
    N = X.shape[0] 
    a = torch.empty(N, device=X.device, dtype=torch.long) 
    c2 = (C * C).sum(dim=1)
    Ct = C.t().contiguous()
    batch = max(1, min(int(batch), int(N)))
    i = 0
    while i < N:
        j = min(i + batch, N) 
        try:
            Xb = X[i:j]
            scores = Xb @ Ct
            scores.add_(c2[None, :], alpha=-0.5)
            a[i:j] = scores.argmax(dim=1)
            i = j
        except RuntimeError as exc:
            if _is_cuda_oom(exc) and batch > RVQ_ASSIGN_MIN_BATCH:
                old_batch = batch
                batch = max(RVQ_ASSIGN_MIN_BATCH, batch // 2)
                _clear_cuda_after_oom()
                log_event(f"CUDA OOM in assignment; reducing batch {_format_batch(old_batch)} -> {_format_batch(batch)}")
                continue
            raise
    return a 


@torch.no_grad()
def distortion(X, C, labels): 
    return ((X - C[labels])**2).sum(dim=1).mean()


@torch.no_grad()
def update_centers_safe(X: torch.Tensor, labels: torch.Tensor, K: int, g: torch.Generator | None = None):
    d = X.shape[1]
    C_new = torch.zeros((K, d), device=X.device, dtype=X.dtype)

    for k in range(K):
        mask = (labels == k)
        if mask.any():
            C_new[k] = X[mask].mean(dim=0)
        else:
            if g is None:
                idx = torch.randint(0, X.shape[0], (1,), device=X.device).item()
            else:
                idx = torch.randint(0, X.shape[0], (1,), generator=g, device=X.device).item()
            C_new[k] = X[idx]
    return C_new


@torch.no_grad()
def kmeans_pp_lloyd_batched(
    X: torch.Tensor,
    K: int,
    max_iter: int = 50,
    tol: float = 1e-4,
    batch: int = 65536,
    seed: int = 5,
):
    """
    X: [N, d] float32 (желательно)
    Возвращает: C [K, d], labels [N], final distortion
    """
    C = init_kmeans_pp(X, K, seed=seed)
    batch = autotune_assignment_batch(X=X, C=C, requested_batch=batch)
    assignment_context = _prepare_multigpu_assignment_context(X)

    g = torch.Generator(device=X.device)
    g.manual_seed(seed + 12345)

    prev_D = None
    eps = 1e-12

    for it in tqdm(range(max_iter), desc="kmeans lloyd", leave=False, unit="iter"):
        if assignment_context is not None:
            labels = _assign_multigpu_from_context(C, batch=batch, context=assignment_context, weighted=False)
        else:
            labels = assign_to_centroids_batched(X, C, batch=batch)
        C = update_centers_safe(X, labels, K, g=g)
        D = distortion(X, C, labels)

        if prev_D is not None:
            rel = (prev_D - D).abs() / max(prev_D.abs(), torch.tensor(eps, device=X.device))
            if rel.item() < tol:
                break
        prev_D = D

    return C, labels, D

# %% cell 15
#=== Обертка для мульти-старта на каждой стадии RVQ, чтобы выбрать лучшую инициализацию ===

@torch.no_grad()
def kmeans_multistart(
    X: torch.Tensor,
    K: int,
    n_init: int,
    base_seed: int,
    max_iter: int,
    tol: float,
    batch: int,
):
    best_C = None
    best_labels = None
    best_D = None

    for r in tqdm(range(n_init), desc="kmeans restarts", leave=False, unit="restart"):
        seed = base_seed + 1000 * r
        C, labels, D = kmeans_pp_lloyd_batched(
            X, K=K, max_iter=max_iter, tol=tol, batch=batch, seed=seed
        )
        if best_D is None or D.item() < best_D.item():
            best_C, best_labels, best_D = C, labels, D

    return best_C, best_labels, best_D


# === Activation-aware k-means: минимизируем (x-c)^T XTX_block (x-c) ===

@torch.no_grad()
def assign_to_centroids_batched_weighted(
    X: torch.Tensor,
    C: torch.Tensor,
    H_blocks: torch.Tensor,
    block_ids: torch.Tensor,
    batch: int = 8192,
):
    N = X.shape[0]
    labels = torch.empty(N, device=X.device, dtype=torch.long)

    batch = max(1, min(int(batch), int(N)))
    C_work = C.to(dtype=X.dtype)
    H_work = H_blocks.to(device=X.device, dtype=X.dtype)
    cHc_blocks = torch.einsum("kd,bdh,kh->bk", C_work, H_work, C_work).contiguous()

    i = 0
    while i < N:
        j = min(i + batch, N)
        try:
            Xb = X[i:j]
            dist2 = _weighted_dist2_no_diff(
                Xb,
                C_work,
                H_blocks=H_work,
                block_ids_b=block_ids[i:j],
                cHc_blocks=cHc_blocks,
                include_xHx=False,
            )
            labels[i:j] = dist2.argmin(dim=1)
            i = j
        except RuntimeError as exc:
            if _is_cuda_oom(exc) and batch > RVQ_ASSIGN_MIN_BATCH:
                old_batch = batch
                batch = max(RVQ_ASSIGN_MIN_BATCH, batch // 2)
                _clear_cuda_after_oom()
                log_event(f"CUDA OOM in weighted assignment; reducing batch {_format_batch(old_batch)} -> {_format_batch(batch)}")
                continue
            raise

    return labels


@torch.no_grad()
def distortion_weighted(
    X: torch.Tensor,
    C: torch.Tensor,
    labels: torch.Tensor,
    H_blocks: torch.Tensor,
    block_ids: torch.Tensor,
):
    diff = X - C[labels]
    H = H_blocks[block_ids]
    values = torch.einsum("bd,bdh,bh->b", diff, H, diff)
    return values.mean()


@torch.no_grad()
def update_centers_weighted_safe(
    X: torch.Tensor,
    labels: torch.Tensor,
    K: int,
    H_blocks: torch.Tensor,
    block_ids: torch.Tensor,
    damping: float = 1e-6,
    g: torch.Generator | None = None,
):
    d = X.shape[1]
    eye = torch.eye(d, device=X.device, dtype=X.dtype)
    C_new = torch.zeros((K, d), device=X.device, dtype=X.dtype)

    # Keep this deterministic: CUDA atomic reductions change run-to-run sums here.
    for k in range(K):
        mask = labels == k
        if mask.any():
            Hk = H_blocks[block_ids[mask]].to(dtype=X.dtype)
            Xk = X[mask]
            A = Hk.sum(dim=0)
            diag_mean = A.diagonal().mean().clamp_min(1e-12)
            A = A + (damping * diag_mean) * eye
            b = torch.einsum("bdh,bh->d", Hk, Xk)
            C_new[k] = torch.linalg.solve(A, b)
        else:
            if g is None:
                idx = torch.randint(0, X.shape[0], (1,), device=X.device).item()
            else:
                idx = torch.randint(0, X.shape[0], (1,), generator=g, device=X.device).item()
            C_new[k] = X[idx]

    return C_new


@torch.no_grad()
def kmeans_pp_lloyd_batched_weighted(
    X: torch.Tensor,
    K: int,
    H_blocks: torch.Tensor,
    block_ids: torch.Tensor,
    max_iter: int = 50,
    tol: float = 1e-4,
    batch: int = 8192,
    seed: int = 5,
    damping: float = 1e-6,
):
    C = init_kmeans_pp(X, K, seed=seed)
    batch = autotune_assignment_batch(
        X=X,
        C=C,
        requested_batch=batch,
        H_blocks=H_blocks,
        block_ids=block_ids,
    )
    assignment_context = _prepare_multigpu_assignment_context(
        X,
        H_blocks=H_blocks,
        block_ids=block_ids,
    )

    g = torch.Generator(device=X.device)
    g.manual_seed(seed + 12345)

    prev_D = None
    eps = torch.tensor(1e-12, device=X.device, dtype=X.dtype)

    for _ in tqdm(range(max_iter), desc="weighted kmeans", leave=False, unit="iter"):
        if assignment_context is not None:
            labels = _assign_multigpu_from_context(C, batch=batch, context=assignment_context, weighted=True)
        else:
            labels = assign_to_centroids_batched_weighted(
                X, C, H_blocks=H_blocks, block_ids=block_ids, batch=batch
            )
        C = update_centers_weighted_safe(
            X, labels, K, H_blocks=H_blocks, block_ids=block_ids, damping=damping, g=g
        )
        D = distortion_weighted(X, C, labels, H_blocks=H_blocks, block_ids=block_ids)

        if prev_D is not None:
            rel = (prev_D - D).abs() / torch.maximum(prev_D.abs(), eps)
            if rel.item() < tol:
                break
        prev_D = D

    return C, labels, D


@torch.no_grad()
def kmeans_multistart_weighted(
    X: torch.Tensor,
    K: int,
    H_blocks: torch.Tensor,
    block_ids: torch.Tensor,
    n_init: int,
    base_seed: int,
    max_iter: int,
    tol: float,
    batch: int,
    damping: float,
):
    best_C = None
    best_labels = None
    best_D = None

    for r in tqdm(range(n_init), desc="weighted restarts", leave=False, unit="restart"):
        seed = base_seed + 1000 * r
        C, labels, D = kmeans_pp_lloyd_batched_weighted(
            X,
            K=K,
            H_blocks=H_blocks,
            block_ids=block_ids,
            max_iter=max_iter,
            tol=tol,
            batch=batch,
            seed=seed,
            damping=damping,
        )
        if best_D is None or D.item() < best_D.item():
            best_C, best_labels, best_D = C, labels, D

    return best_C, best_labels, best_D

# %% [markdown] cell 16
# <h1>Квантизация нейронки по слоям (сначала отдельно слой, потом как погружаться вниз по слоям)</h1>

# %% cell 17
# === Compressed RVQ Linear ===


def _rvq_code_storage_dtype(K: int) -> torch.dtype:
    if K <= 256:
        return torch.uint8
    if K <= 32768:
        return torch.int16
    return torch.int32


def _rvq_ceil_log2_positive(n: int) -> int:
    n = max(1, int(n))
    return int(math.ceil(math.log2(n))) if n > 1 else 1


def _pad_rvq_codebooks(codebooks: Sequence[torch.Tensor], stage_ks: Sequence[int]) -> torch.Tensor:
    if len(codebooks) != len(stage_ks):
        raise ValueError(f"codebooks/stage_ks length mismatch: {len(codebooks)} vs {len(stage_ks)}")
    max_k = max(int(k) for k in stage_ks)
    padded = []
    for C, K in zip(codebooks, stage_ks):
        K = int(K)
        if int(C.shape[0]) != K:
            raise ValueError(f"Codebook shape {tuple(C.shape)} does not match stage K={K}")
        if K < max_k:
            pad = torch.zeros((max_k - K, int(C.shape[1])), device=C.device, dtype=C.dtype)
            C = torch.cat([C, pad], dim=0)
        padded.append(C)
    return torch.stack(padded, dim=0)


def _rvq_stage_ks_from_codebooks(cfg: VQConfig, codebooks: torch.Tensor) -> Tuple[int, ...]:
    stage_ks = _cfg_stage_ks(cfg)
    M = int(codebooks.shape[0])
    if len(stage_ks) != M:
        if len(stage_ks) > M:
            stage_ks = tuple(stage_ks[:M])
        else:
            stage_ks = tuple(stage_ks) + tuple(int(codebooks.shape[1]) for _ in range(M - len(stage_ks)))
    max_k = int(codebooks.shape[1])
    return tuple(min(int(k), max_k) for k in stage_ks)


def compute_rvq_code_usage_stats(
    codes: torch.Tensor,
    stage_ks: Sequence[int],
    *,
    d: int,
) -> Dict[str, Any]:
    codes_cpu = codes.detach().to("cpu", dtype=torch.long)
    if codes_cpu.ndim < 2:
        codes_cpu = codes_cpu.reshape(int(codes_cpu.shape[0]), -1)
    else:
        codes_cpu = codes_cpu.reshape(int(codes_cpu.shape[0]), -1)

    stage_ks = tuple(int(k) for k in stage_ks)
    if len(stage_ks) != int(codes_cpu.shape[0]):
        raise ValueError(f"codes/stage_ks length mismatch: {codes_cpu.shape[0]} vs {len(stage_ks)}")

    per_stage = []
    total_entropy = 0.0
    total_bits = 0
    total_used = 0
    total_dead = 0
    total_codes = 0
    invalid_codes = 0
    for m, K in enumerate(stage_ks):
        flat = codes_cpu[m].reshape(-1)
        invalid = int(((flat < 0) | (flat >= K)).sum().item())
        invalid_codes += invalid
        flat_valid = flat[(flat >= 0) & (flat < K)]
        counts = torch.bincount(flat_valid, minlength=K).to(torch.float64)
        n = float(counts.sum().item())
        used = int((counts > 0).sum().item())
        probs = counts[counts > 0] / max(n, 1.0)
        entropy = float((-(probs * torch.log2(probs))).sum().item()) if probs.numel() else 0.0
        stage_bits = _rvq_ceil_log2_positive(K)
        per_stage.append({
            "stage": int(m),
            "K": int(K),
            "bits": int(stage_bits),
            "n_codes": int(n),
            "used_codes": int(used),
            "dead_codes": int(K - used),
            "utilization": float(used / K),
            "entropy": entropy,
            "perplexity": float(2.0 ** entropy),
            "normalized_entropy": float(entropy / stage_bits) if stage_bits > 0 else 0.0,
            "max_code_probability": float((counts.max() / max(n, 1.0)).item()) if counts.numel() else 0.0,
            "invalid_codes": int(invalid),
        })
        total_entropy += entropy
        total_bits += stage_bits
        total_used += used
        total_dead += K - used
        total_codes += K

    return {
        "K": int(max(stage_ks)),
        "bits": int(max(_stage_bits_for_ks(stage_ks))),
        "stage_Ks": list(stage_ks),
        "stage_bits": list(_stage_bits_for_ks(stage_ks)),
        "d": int(d),
        "packed_index_bpp": float(total_bits / max(int(d), 1)),
        "entropy_bpp": float(total_entropy / max(int(d), 1)),
        "entropy_bits_per_vector": float(total_entropy),
        "packed_bits_per_vector": int(total_bits),
        "used_codes_total": int(total_used),
        "dead_codes_total": int(total_dead),
        "codebook_entries_total": int(total_codes),
        "mean_utilization": float(total_used / total_codes) if total_codes else 0.0,
        "mean_normalized_entropy": float(sum(s["normalized_entropy"] for s in per_stage) / len(per_stage)) if per_stage else 0.0,
        "invalid_codes": int(invalid_codes),
        "per_stage": per_stage,
    }


@torch.no_grad()
def apply_hadamard_lastdim(
    x: torch.Tensor,
    block: int,
    seed: int,
    with_signs: bool = True,
    inverse: bool = False,
) -> torch.Tensor:
    """Apply the same block Hadamard transform as weights, but along activation last dim."""
    assert (block & (block - 1)) == 0, "block must be power of 2"
    y = x.contiguous().clone()
    last_dim = y.shape[-1]
    i2 = (last_dim // block) * block
    if i2 == 0:
        return y

    y_main = y[..., :i2].contiguous().view(-1, i2 // block, block)

    if not inverse and with_signs:
        s = rademacher_signs(block, device=y.device, seed=seed, dtype=y.dtype)
        y_main.mul_(s)

    fwht_(y_main)
    y_main.mul_(1.0 / math.sqrt(block))

    if inverse and with_signs:
        s = rademacher_signs(block, device=y.device, seed=seed, dtype=y.dtype)
        y_main.mul_(s)

    y[..., :i2] = y_main.view(*y.shape[:-1], i2)
    return y


class QuantizedRVQLinear(nn.Module):
    def __init__(
        self,
        *,
        in_features: int,
        out_features: int,
        codebooks: torch.Tensor,
        codes: torch.Tensor,
        d: int,
        n_chunks: int,
        pad: int,
        bias: Optional[torch.Tensor],
        bits: int,
        use_hadamard: bool,
        hadamard_block: int,
        hadamard_seed: int,
        hadamard_with_signs: bool,
        activation_aware: bool,
        stage_distortions: Sequence[float],
        full_xtx_recode_stats: Optional[Dict[str, float]],
        dtype: torch.dtype,
        device,
        stage_Ks: Optional[Sequence[int]] = None,
        stage_bits: Optional[Sequence[int]] = None,
        code_usage_stats: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.d = int(d)
        self.n_chunks = int(n_chunks)
        self.pad = int(pad)
        self.stage_Ks = tuple(int(k) for k in (stage_Ks or [int(codebooks.shape[1])] * int(codebooks.shape[0])))
        self.stage_bits = tuple(int(b) for b in (stage_bits or _stage_bits_for_ks(self.stage_Ks)))
        self.bits = int(max(self.stage_bits)) if self.stage_bits else int(bits)
        self.rvq_M = int(codebooks.shape[0])
        self.vq_K = int(max(self.stage_Ks)) if self.stage_Ks else int(codebooks.shape[1])
        self.vq_d = int(d)
        self.use_hadamard = bool(use_hadamard)
        self.hadamard_block = int(hadamard_block)
        self.hadamard_seed = int(hadamard_seed)
        self.hadamard_with_signs = bool(hadamard_with_signs)
        self.activation_aware = bool(activation_aware)
        self.is_rvq = True
        self.rvq_compressed = True
        self.rvq_stage_distortions = list(stage_distortions)
        self.rvq_full_xtx_recode_stats = dict(full_xtx_recode_stats or {})
        self.rvq_code_usage_stats = dict(code_usage_stats or {})

        self.register_buffer("codebooks", codebooks.to(device=device, dtype=dtype).contiguous())
        self.register_buffer("codes", codes.to(device=device).contiguous())

        if bias is None:
            self.register_parameter("bias", None)
        else:
            self.bias = nn.Parameter(bias.detach().to(device=device, dtype=dtype).clone(), requires_grad=False)

    def dequantize_rotated_weight(self, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
        dtype = dtype or self.codebooks.dtype
        codebooks = self.codebooks.to(dtype=dtype)
        codes = self.codes.to(device=codebooks.device, dtype=torch.long)

        total_vectors = self.out_features * self.n_chunks
        Xq = torch.zeros(
            (total_vectors, self.d),
            device=codebooks.device,
            dtype=codebooks.dtype,
        )
        for m in range(self.rvq_M):
            K_m = int(self.stage_Ks[m]) if m < len(self.stage_Ks) else int(codebooks.shape[1])
            Xq.add_(codebooks[m, :K_m].index_select(0, codes[m].reshape(-1)))

        W2 = Xq.view(self.out_features, self.n_chunks, self.d).reshape(
            self.out_features,
            self.n_chunks * self.d,
        )
        if self.pad > 0:
            W2 = W2[:, :self.in_features]
        return W2.contiguous()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        weight = self.dequantize_rotated_weight(dtype=input.dtype)
        x = input
        if self.use_hadamard:
            x = apply_hadamard_lastdim(
                x,
                block=self.hadamard_block,
                seed=self.hadamard_seed,
                with_signs=self.hadamard_with_signs,
                inverse=False,
            )
        bias = self.bias.to(dtype=input.dtype) if self.bias is not None else None
        return F.linear(x, weight, bias)

    def estimate_nbits_per_parameter(self, codebook_bits: int = 16, bias_bits: int = 16) -> float:
        weight_params = self.out_features * self.in_features
        index_bits = self.out_features * self.n_chunks * sum(self.stage_bits)
        codebook_bits_total = sum(self.stage_Ks) * self.d * codebook_bits
        bias_bits_total = self.out_features * bias_bits if self.bias is not None else 0
        return (index_bits + codebook_bits_total + bias_bits_total) / weight_params

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"M={self.rvq_M}, Ks={_format_stage_ks(self.stage_Ks)}, d={self.d}, "
            f"compressed_bpp~{self.estimate_nbits_per_parameter():.3f}"
        )


@torch.no_grad()
def _run_rvq_stages(X: torch.Tensor, cfg: VQConfig, layer_salt: int, H_blocks=None, block_ids=None):
    R = X.clone()
    Xq = torch.zeros_like(X)
    all_D = []
    all_codebooks = []
    all_codes = []

    stage_ks = _cfg_stage_ks(cfg)
    M = len(stage_ks)
    code_dtype = _rvq_code_storage_dtype(max(stage_ks))
    stage_iter = range(M)
    if cfg.show_inner_progress:
        stage_iter = tqdm(stage_iter, desc="RVQ stages", leave=cfg.progress_leave, unit="stage")

    for m in stage_iter:
        K_m = int(stage_ks[m])
        base_seed = cfg.seed + 7777 * m + layer_salt
        use_multistart = (cfg.rvq_first_stage_n_init > 1)
        assign_batch = int(cfg.batch_assign or 0)

        if H_blocks is not None and block_ids is not None:
            if use_multistart:
                C, labels, D = kmeans_multistart_weighted(
                    R,
                    K=K_m,
                    H_blocks=H_blocks,
                    block_ids=block_ids,
                    n_init=cfg.rvq_first_stage_n_init,
                    base_seed=base_seed,
                    max_iter=cfg.max_iter,
                    tol=cfg.tol,
                    batch=assign_batch,
                    damping=cfg.xtx_damping,
                )
            else:
                C, labels, D = kmeans_pp_lloyd_batched_weighted(
                    R,
                    K=K_m,
                    H_blocks=H_blocks,
                    block_ids=block_ids,
                    max_iter=cfg.max_iter,
                    tol=cfg.tol,
                    batch=assign_batch,
                    seed=base_seed,
                    damping=cfg.xtx_damping,
                )
        else:
            if use_multistart:
                C, labels, D = kmeans_multistart(
                    R,
                    K=K_m,
                    n_init=cfg.rvq_first_stage_n_init,
                    base_seed=base_seed,
                    max_iter=cfg.max_iter,
                    tol=cfg.tol,
                    batch=assign_batch,
                )
            else:
                C, labels, D = kmeans_pp_lloyd_batched(
                    R,
                    K=K_m,
                    max_iter=cfg.max_iter,
                    tol=cfg.tol,
                    batch=assign_batch,
                    seed=base_seed,
                )

        all_codebooks.append(C.detach().cpu())
        all_codes.append(labels.detach().to("cpu", dtype=code_dtype))

        Qm = C[labels]
        Xq.add_(Qm)
        R.sub_(Qm)

        all_D.append(D.detach().cpu())
        del Qm, C, labels
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if cfg.show_inner_progress and hasattr(stage_iter, "set_postfix"):
            stage_iter.set_postfix(loss=f"{D.item():.4g}", K=K_m)

    codebooks = _pad_rvq_codebooks(all_codebooks, stage_ks)
    codes = torch.stack(all_codes, dim=0)
    return Xq, all_D, codebooks, codes


@torch.no_grad()
def _rvq_full_xtx_objective(
    X: torch.Tensor,
    Xq: torch.Tensor,
    H_full: Optional[torch.Tensor],
    meta: VectorizeMeta,
) -> float:
    if H_full is None:
        return float("nan")
    rows = meta.prefix_shape[0]
    padded_dim = meta.n_chunks * meta.d
    diff = (Xq - X).reshape(rows, padded_dim)
    H = H_full.to(device=diff.device, dtype=diff.dtype)
    return float(((diff @ H) * diff).sum(dim=1).mean().item())


@torch.no_grad()
def refine_rvq_codes_full_xtx(
    X: torch.Tensor,
    Xq: torch.Tensor,
    codebooks: torch.Tensor,
    codes: torch.Tensor,
    H_full: Optional[torch.Tensor],
    meta: VectorizeMeta,
    cfg: VQConfig,
    layer_salt: int,
):
    """
    Coordinate-descent post pass with fixed RVQ codebooks and full layer XTX.

    RVQ training uses independent d-sized Hessian blocks. This pass keeps the learned
    codebooks fixed, but updates integer codes while scoring each local replacement
    against the full padded XTX, so cross-chunk interactions can improve the final
    activation-aware objective without running full AQ/beam-search training.
    """
    stats = {
        "enabled": float(H_full is not None),
        "skipped": 0.0,
        "before": float("nan"),
        "after": float("nan"),
        "delta": float("nan"),
        "updates": 0.0,
        "passes": 0.0,
    }
    if H_full is None:
        stats["skipped"] = 1.0
        return Xq, codes, stats

    max_cols = int(cfg.full_xtx_recode_max_cols or 0)
    if max_cols > 0 and int(meta.last_dim) > max_cols:
        stats["skipped"] = 1.0
        log_event(
            f"[RVQ full-XTX recode] skipped layer_salt={layer_salt}: "
            f"last_dim={meta.last_dim} > max_cols={max_cols}"
        )
        return Xq, codes, stats

    passes = max(int(cfg.full_xtx_recode_passes or 1), 1)

    rows = int(meta.prefix_shape[0])
    n_chunks = int(meta.n_chunks)
    d = int(meta.d)
    padded_dim = n_chunks * d
    M = int(codebooks.shape[0])
    stage_ks = _rvq_stage_ks_from_codebooks(cfg, codebooks)
    row_batch = int(cfg.full_xtx_recode_row_batch or rows)
    row_batch = max(1, min(row_batch, rows))

    H = H_full.to(device=X.device, dtype=X.dtype).contiguous()
    C_all = codebooks.to(device=X.device, dtype=X.dtype).contiguous()
    codes_work = codes.to(device=X.device, dtype=torch.long).reshape(M, rows, n_chunks).clone()

    X_rows = X.reshape(rows, padded_dim)
    Xq_rows = Xq.reshape(rows, padded_dim).clone()
    E = Xq_rows - X_rows
    G = E @ H

    before = float((G * E).sum(dim=1).mean().item())
    total_updates = 0

    pass_iter = range(passes)
    if cfg.show_inner_progress:
        pass_iter = tqdm(pass_iter, desc="full-XTX recode", leave=False, unit="pass")

    for _ in pass_iter:
        pass_updates = 0
        for chunk in range(n_chunks):
            sl = slice(chunk * d, (chunk + 1) * d)
            Hjj = H[sl, sl].contiguous()
            H_row = H[sl, :].contiguous()

            for m in range(M):
                C = C_all[m, : int(stage_ks[m])]
                old_ids_all = codes_work[m, :, chunk]

                for start in range(0, rows, row_batch):
                    stop = min(start + row_batch, rows)
                    old_ids = old_ids_all[start:stop]
                    old_vec = C.index_select(0, old_ids)
                    diff = C[None, :, :] - old_vec[:, None, :]

                    linear_delta = 2.0 * torch.einsum("bkd,bd->bk", diff, G[start:stop, sl])
                    quad_delta = torch.einsum("bkd,dh,bkh->bk", diff, Hjj, diff)
                    delta = linear_delta + quad_delta
                    best_delta, best_ids = delta.min(dim=1)
                    improve = best_delta < -1e-8
                    if not improve.any():
                        continue

                    row_ids = torch.arange(start, stop, device=X.device, dtype=torch.long)[improve]
                    chosen_diff = C.index_select(0, best_ids[improve]) - old_vec[improve]
                    E[row_ids, sl] += chosen_diff
                    G[row_ids, :] += chosen_diff @ H_row
                    codes_work[m, row_ids, chunk] = best_ids[improve]
                    n_changed = int(improve.sum().item())
                    pass_updates += n_changed
                    total_updates += n_changed

        if cfg.show_inner_progress and hasattr(pass_iter, "set_postfix"):
            pass_iter.set_postfix(updates=pass_updates)
        if pass_updates == 0:
            break

    after = float((G * E).sum(dim=1).mean().item())
    Xq_refined = (X_rows + E).reshape_as(Xq)
    codes_refined = codes_work.reshape(M, -1).detach().cpu().to(dtype=_rvq_code_storage_dtype(max(stage_ks)))

    stats.update({
        "skipped": 0.0,
        "before": before,
        "after": after,
        "delta": after - before,
        "updates": float(total_updates),
        "passes": float(passes),
    })
    log_event(
        f"[RVQ full-XTX recode] layer_salt={layer_salt} "
        f"before={before:.6g}, after={after:.6g}, delta={after - before:.6g}, "
        f"updates={total_updates}, passes={passes}, rows={rows}, chunks={n_chunks}, M={M}"
    )
    return Xq_refined, codes_refined, stats


@torch.no_grad()
def quantize_linear_layer_rvq(
    linear: nn.Linear,
    cfg: VQConfig,
    layer_salt: int = 0,
    xtx: Optional[torch.Tensor] = None,
):
    """
    RVQ for Linear weights: split the last dimension into d-vectors and quantize residuals M times.
    If cfg.use_activation_aware=True and xtx is provided, assignments/centers use XTX blocks.
    If cfg.store_compressed=True, return QuantizedRVQLinear with codes+codebooks.
    """
    w = linear.weight.data
    orig_device = w.device
    orig_dtype = w.dtype

    if cfg.on_cpu:
        w_work = w.detach().to("cpu", dtype=torch.float32)
    else:
        w_work = w.detach().to(orig_device, dtype=torch.float32)

    had_seed = cfg.hadamard_seed + cfg.seed + layer_salt
    if cfg.use_hadamard:
        hadamard_right_blocks(
            w_work,
            block=cfg.hadamard_block,
            seed=had_seed,
            with_signs=cfg.hadamard_with_signs,
        )

    X, meta = weight_to_vectors_lastdim(w_work, cfg.d)
    H_blocks, block_ids, H_full = make_vector_hessian_blocks(
        xtx=xtx,
        meta=meta,
        cfg=cfg,
        layer_salt=layer_salt,
        device=X.device,
    )

    Xq, all_D, codebooks, codes = _run_rvq_stages(
        X, cfg=cfg, layer_salt=layer_salt, H_blocks=H_blocks, block_ids=block_ids
    )
    Xq, codes, full_xtx_recode_stats = refine_rvq_codes_full_xtx(
        X,
        Xq,
        codebooks,
        codes,
        H_full,
        meta,
        cfg,
        layer_salt,
    )

    stage_distortions = [float(d.item()) for d in all_D]
    activation_aware = bool(cfg.use_activation_aware and xtx is not None)
    stage_ks = _cfg_stage_ks(cfg)
    stage_bits = _stage_bits_for_ks(stage_ks)
    code_usage_stats = compute_rvq_code_usage_stats(codes, stage_ks, d=cfg.d)

    if cfg.store_compressed:
        return QuantizedRVQLinear(
            in_features=linear.in_features,
            out_features=linear.out_features,
            codebooks=codebooks,
            codes=codes.view(len(stage_ks), meta.prefix_shape[0], meta.n_chunks),
            d=cfg.d,
            n_chunks=meta.n_chunks,
            pad=meta.pad,
            bias=linear.bias.data if linear.bias is not None else None,
            bits=cfg.bits,
            use_hadamard=cfg.use_hadamard,
            hadamard_block=cfg.hadamard_block,
            hadamard_seed=had_seed,
            hadamard_with_signs=cfg.hadamard_with_signs,
            activation_aware=activation_aware,
            stage_distortions=stage_distortions,
            full_xtx_recode_stats=full_xtx_recode_stats,
            dtype=orig_dtype,
            device=orig_device,
            stage_Ks=stage_ks,
            stage_bits=stage_bits,
            code_usage_stats=code_usage_stats,
        )

    new_linear = nn.Linear(
        linear.in_features,
        linear.out_features,
        bias=(linear.bias is not None),
    ).to(device=linear.weight.device, dtype=linear.weight.dtype)

    wq_rot = vectors_to_weight_lastdim(Xq, meta, dtype=torch.float32)

    if cfg.use_hadamard:
        hadamard_right_blocks_inv(
            wq_rot,
            block=cfg.hadamard_block,
            seed=had_seed,
            with_signs=cfg.hadamard_with_signs,
        )

    wq = wq_rot.to(device=orig_device, dtype=orig_dtype)
    new_linear.weight.data.copy_(wq)

    if linear.bias is not None:
        new_linear.bias.data.copy_(linear.bias.data)

    new_linear.is_rvq = True
    new_linear.rvq_compressed = False
    new_linear.rvq_M = len(stage_ks)
    new_linear.vq_K = max(stage_ks)
    new_linear.stage_Ks = tuple(stage_ks)
    new_linear.stage_bits = tuple(stage_bits)
    new_linear.bits = max(stage_bits)
    new_linear.vq_d = cfg.d
    new_linear.vq_seed = cfg.seed
    new_linear.activation_aware = activation_aware
    new_linear.rvq_stage_distortions = stage_distortions
    new_linear.rvq_full_xtx_recode_stats = dict(full_xtx_recode_stats or {})
    new_linear.rvq_code_usage_stats = dict(code_usage_stats or {})

    return new_linear

# %% cell 18
# === Векторная квантизация Conv ===

@torch.no_grad()
def quantize_conv_layer_rvq(
    conv: nn.Conv2d,
    cfg: VQConfig,
    layer_salt: int = 0,
    xtx: Optional[torch.Tensor] = None,
):
    """
    RVQ по весам Conv2d.
    При cfg.group_channels=True используем VQDM-style flatten:
    [out, in, kh, kw] -> [out, kh, kw, in] -> [out, kh*kw*in].
    """
    new_conv = nn.Conv2d(
        conv.in_channels,
        conv.out_channels,
        kernel_size=conv.kernel_size,
        stride=conv.stride,
        padding=conv.padding,
        dilation=conv.dilation,
        groups=conv.groups,
        bias=(conv.bias is not None),
        padding_mode=conv.padding_mode,
    ).to(device=conv.weight.device, dtype=conv.weight.dtype)

    w = conv.weight.data
    orig_device = w.device
    orig_dtype = w.dtype

    if cfg.on_cpu:
        w_work = w.detach().to("cpu", dtype=torch.float32)
    else:
        w_work = w.detach().to(orig_device, dtype=torch.float32)

    if cfg.group_channels:
        w_work = w_work.permute(0, 2, 3, 1).contiguous()

    if cfg.use_hadamard:
        hadamard_right_blocks(
            w_work,
            block=cfg.hadamard_block,
            seed=cfg.hadamard_seed + cfg.seed + layer_salt,
            with_signs=cfg.hadamard_with_signs,
        )

    X, meta = weight_to_vectors_lastdim(w_work, cfg.d)
    H_blocks, block_ids, H_full = make_vector_hessian_blocks(
        xtx=xtx,
        meta=meta,
        cfg=cfg,
        layer_salt=layer_salt,
        device=X.device,
    )

    Xq, all_D, _codebooks, _codes = _run_rvq_stages(
        X, cfg=cfg, layer_salt=layer_salt, H_blocks=H_blocks, block_ids=block_ids
    )
    Xq, _codes, full_xtx_recode_stats = refine_rvq_codes_full_xtx(
        X,
        Xq,
        _codebooks,
        _codes,
        H_full,
        meta,
        cfg,
        layer_salt,
    )
    stage_ks = _cfg_stage_ks(cfg)
    stage_bits = _stage_bits_for_ks(stage_ks)
    code_usage_stats = compute_rvq_code_usage_stats(_codes, stage_ks, d=cfg.d)

    wq_rot = vectors_to_weight_lastdim(Xq, meta, dtype=torch.float32)

    if cfg.use_hadamard:
        hadamard_right_blocks_inv(
            wq_rot,
            block=cfg.hadamard_block,
            seed=cfg.hadamard_seed + cfg.seed + layer_salt,
            with_signs=cfg.hadamard_with_signs,
        )

    if cfg.group_channels:
        wq_rot = wq_rot.permute(0, 3, 1, 2).contiguous()

    wq = wq_rot.to(device=orig_device, dtype=orig_dtype)
    new_conv.weight.data.copy_(wq)

    if conv.bias is not None:
        new_conv.bias.data.copy_(conv.bias.data)

    new_conv.is_rvq = True
    new_conv.rvq_compressed = False
    new_conv.rvq_M = len(stage_ks)
    new_conv.vq_K = max(stage_ks)
    new_conv.stage_Ks = tuple(stage_ks)
    new_conv.stage_bits = tuple(stage_bits)
    new_conv.bits = max(stage_bits)
    new_conv.vq_d = cfg.d
    new_conv.vq_seed = cfg.seed
    new_conv.activation_aware = bool(cfg.use_activation_aware and xtx is not None)
    new_conv.rvq_stage_distortions = [float(d.item()) for d in all_D]
    new_conv.rvq_full_xtx_recode_stats = dict(full_xtx_recode_stats or {})
    new_conv.rvq_code_usage_stats = dict(code_usage_stats or {})

    return new_conv

# %% cell 19
# === Рекурсивная квантизация выбранных Conv2d и Linear внутри модуля ===

def stable_layer_salt(name: str) -> int:
    digest = hashlib.sha1(name.encode("utf-8")).hexdigest()[:8]
    return int(digest, 16) & 0x7fffffff


def _get_xtx_for_layer(xtx_stats: Optional[Dict[str, Any]], full_name: str):
    if xtx_stats is None or full_name not in xtx_stats:
        return None
    item = xtx_stats[full_name]
    if isinstance(item, dict) and "xtx" in item:
        return item["xtx"]
    return item


def _release_layer_xtx(xtx_stats: Optional[Dict[str, Any]], full_name: str, cfg: VQConfig):
    if not cfg.free_xtx_after_layer or xtx_stats is None or full_name not in xtx_stats:
        return
    item = xtx_stats.get(full_name)
    if isinstance(item, dict) and "xtx" in item:
        item["xtx"] = None
    else:
        xtx_stats[full_name] = None
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def quantize_module_weights(
    module: nn.Module,
    cfg: VQConfig = VQC,
    pbar: tqdm | None = None,
    prefix: str = "",
    xtx_stats: Optional[Dict[str, Any]] = None,
    target_layer_names: Optional[set[str]] = None,
) -> nn.Module:
    for name, child in module.named_children():
        full = f"{prefix}.{name}" if prefix else name

        should_quantize = target_layer_names is None or full in target_layer_names

        if isinstance(child, nn.Linear) and should_quantize:
            t0 = time.perf_counter()
            new = quantize_linear_layer_rvq(
                child,
                cfg=cfg,
                layer_salt=stable_layer_salt(full),
                xtx=_get_xtx_for_layer(xtx_stats, full),
            )
            if not cfg.on_cpu and torch.cuda.is_available():
                torch.cuda.synchronize()
            dt = time.perf_counter() - t0
            setattr(module, name, new)
            _release_layer_xtx(xtx_stats, full, cfg)

            if pbar is not None:
                pbar.update(1)
                suffix = " act" if getattr(new, "activation_aware", False) else " weight"
                storage = " compressed" if getattr(new, "rvq_compressed", False) else " dense"
                pbar.set_postfix_str(f"Linear {tuple(child.weight.shape)} | {dt:.2f}s |{suffix}{storage}")
                pbar.refresh()

        elif isinstance(child, nn.Conv2d) and should_quantize:
            t0 = time.perf_counter()
            new = quantize_conv_layer_rvq(
                child,
                cfg=cfg,
                layer_salt=stable_layer_salt(full),
                xtx=_get_xtx_for_layer(xtx_stats, full),
            )
            if not cfg.on_cpu and torch.cuda.is_available():
                torch.cuda.synchronize()
            dt = time.perf_counter() - t0
            setattr(module, name, new)
            _release_layer_xtx(xtx_stats, full, cfg)

            if pbar is not None:
                pbar.update(1)
                suffix = " act" if getattr(new, "activation_aware", False) else " weight"
                pbar.set_postfix_str(f"Conv2d {tuple(child.weight.shape)} | {dt:.2f}s |{suffix}")
                pbar.refresh()

        else:
            quantize_module_weights(
                child,
                cfg=cfg,
                pbar=pbar,
                prefix=full,
                xtx_stats=xtx_stats,
                target_layer_names=target_layer_names,
            )

    return module

# %% cell 20
# === Квантизация слоев в модуле ===

def count_target_quantizable_layers(module: nn.Module, prefix: str = "", target_layer_names: Optional[set[str]] = None):
    n = 0
    for name, child in module.named_modules():
        if name == "":
            continue
        full = f"{prefix}.{name}" if prefix else name
        if isinstance(child, (nn.Linear, nn.Conv2d)) and (target_layer_names is None or full in target_layer_names):
            n += 1
    return n


def quantize_unet_all_weights(
    unet: nn.Module,
    cfg: VQConfig = VQC,
    xtx_stats: Optional[Dict[str, Any]] = None,
    target_layer_names: Optional[set[str]] = None,
    prefix: str = "",
    final_device: str = "cuda",
    final_dtype: torch.dtype = torch.float16,
    cpu_deepcopy_before_quant: bool = False,
) -> nn.Module:
    work_device = "cpu" if cfg.on_cpu else final_device
    work_dtype = torch.float32 if cfg.on_cpu else final_dtype

    # Low-VRAM legacy path: offload the source block before deepcopy to reduce peak GPU memory.
    # On high-VRAM GPUs keep everything on CUDA to avoid slow CPU<->GPU copies.
    first_param = next(unet.parameters(), None)
    source_device = first_param.device if first_param is not None else torch.device("cpu")
    if cpu_deepcopy_before_quant and source_device.type != "cpu":
        print(f"[RVQ] Moving source block from {source_device} to CPU before deepcopy (legacy low-VRAM path)")
        unet.to("cpu")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    elif str(source_device) != str(work_device):
        log_event(f"[RVQ] Keeping source block on {source_device} for deepcopy; work_device={work_device}")

    new_unet = copy.deepcopy(unet).to(work_device, dtype=work_dtype)

    total = count_target_quantizable_layers(new_unet, prefix=prefix, target_layer_names=target_layer_names)
    mode = "activation-aware" if cfg.use_activation_aware and xtx_stats is not None else "weight-only"
    print(f"[RVQ] Quantizing {total} Linear/Conv2d layers ({mode})")

    with tqdm(total=total, desc="RVQ-UNet", unit="layer") as pbar:
        quantize_module_weights(
            new_unet,
            cfg=cfg,
            pbar=pbar,
            prefix=prefix,
            xtx_stats=xtx_stats,
            target_layer_names=target_layer_names,
        )

    new_unet = new_unet.to(final_device, dtype=final_dtype)
    summarize_rvq_compressed_modules(new_unet, verbose=True)
    return new_unet


def summarize_rvq_compressed_modules(module: nn.Module, verbose: bool = True) -> Dict[str, Any]:
    rows = []
    total_weight_params = 0
    total_index_bits = 0
    total_codebook_bits = 0
    total_bias_bits = 0

    for name, child in module.named_modules():
        if not isinstance(child, QuantizedRVQLinear):
            continue

        weight_params = child.out_features * child.in_features
        stage_bits = tuple(getattr(child, "stage_bits", []) or [child.bits] * child.rvq_M)
        stage_ks = tuple(getattr(child, "stage_Ks", []) or [child.vq_K] * child.rvq_M)
        index_bits = child.out_features * child.n_chunks * sum(stage_bits)
        codebook_bits = sum(stage_ks) * child.d * 16
        bias_bits = child.out_features * 16 if child.bias is not None else 0

        total_weight_params += weight_params
        total_index_bits += index_bits
        total_codebook_bits += codebook_bits
        total_bias_bits += bias_bits

        rows.append({
            "name": name,
            "weight_params": weight_params,
            "index_bits": index_bits,
            "codebook_bits": codebook_bits,
            "bias_bits": bias_bits,
            "bpp_including_bias": (index_bits + codebook_bits + bias_bits) / weight_params,
        })

    total_bits = total_index_bits + total_codebook_bits + total_bias_bits
    result = {
        "n_layers": len(rows),
        "weight_params": total_weight_params,
        "index_bits": total_index_bits,
        "codebook_bits": total_codebook_bits,
        "bias_bits": total_bias_bits,
        "total_bits": total_bits,
        "bpp_including_bias": total_bits / total_weight_params if total_weight_params else 0.0,
        "rows": rows,
    }

    if verbose and rows:
        print("=== Compressed RVQ storage (QuantizedRVQLinear) ===")
        print(f"Compressed RVQ Linear layers : {result['n_layers']:,}")
        print(f"Target weight params         : {result['weight_params']:,}")
        print(f"Index bits                   : {result['index_bits']:,}")
        print(f"Codebook bits                : {result['codebook_bits']:,}")
        print(f"Bias bits                    : {result['bias_bits']:,}")
        print(f"Total stored bits            : {result['total_bits']:,}")
        print(f"Stored bpp incl. bias        : {result['bpp_including_bias']:.6f}")

    return result


@torch.no_grad()
def materialize_quantized_rvq_linear_for_inference(layer: QuantizedRVQLinear, device=None, dtype=None) -> nn.Linear:
    """
    Convert compressed QuantizedRVQLinear to a dense nn.Linear for fast diffusion inference.

    This preserves the mathematical forward pass, including Hadamard variants, but avoids
    reconstructing the dense weight from RVQ codes on every UNet forward call.
    """
    device = device or next(layer.buffers()).device
    dtype = dtype or layer.codebooks.dtype

    w = layer.dequantize_rotated_weight(dtype=torch.float32)
    if layer.use_hadamard:
        hadamard_right_blocks_inv(
            w,
            block=layer.hadamard_block,
            seed=layer.hadamard_seed,
            with_signs=layer.hadamard_with_signs,
        )

    dense = nn.Linear(layer.in_features, layer.out_features, bias=(layer.bias is not None))
    dense = dense.to(device=device, dtype=dtype)
    dense.weight.data.copy_(w.to(device=device, dtype=dtype))
    if layer.bias is not None:
        dense.bias.data.copy_(layer.bias.detach().to(device=device, dtype=dtype))

    dense.is_rvq = True
    dense.rvq_compressed = False
    dense.rvq_materialized_for_eval = True
    dense.rvq_M = layer.rvq_M
    dense.vq_K = layer.vq_K
    dense.vq_d = layer.vq_d
    dense.bits = layer.bits
    dense.use_hadamard = layer.use_hadamard
    dense.activation_aware = layer.activation_aware
    dense.rvq_stage_distortions = list(layer.rvq_stage_distortions)
    dense.rvq_full_xtx_recode_stats = dict(getattr(layer, "rvq_full_xtx_recode_stats", {}) or {})
    dense.rvq_code_usage_stats = dict(getattr(layer, "rvq_code_usage_stats", {}) or {})
    return dense


@torch.no_grad()
def materialize_compressed_rvq_linears_for_inference(module: nn.Module, device=None, dtype=None) -> int:
    n = 0
    for name, child in list(module.named_children()):
        if isinstance(child, QuantizedRVQLinear):
            setattr(module, name, materialize_quantized_rvq_linear_for_inference(child, device=device, dtype=dtype))
            n += 1
        else:
            n += materialize_compressed_rvq_linears_for_inference(child, device=device, dtype=dtype)
    return n

# %% [markdown] cell 21
# <h1>Всякие метрики и визуализация сравнения квантизованной и оригинальной нейронок</h1>

# %% cell 22
# === Metrics ===

def _module_device(module: nn.Module):
    return next(module.parameters()).device


def prepare_image_for_clip(img):
    return clip_preprocess(img).unsqueeze(0).to(_module_device(clip_model))


def compute_clip_text_image_score(model_clip, tokenizer, text: str, img_tensor):
    dev = _module_device(model_clip)
    txt = tokenizer([text]).to(dev)
    with torch.no_grad():
        txt_emb = model_clip.encode_text(txt)
        img_emb = model_clip.encode_image(img_tensor.to(dev))

    txt_emb = txt_emb / txt_emb.norm(dim=-1, keepdim=True)
    img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)

    return (txt_emb @ img_emb.T).item()


def compute_clip_image_image_score(model_clip, img_a, img_b):
    dev = _module_device(model_clip)
    with torch.no_grad():
        e1 = model_clip.encode_image(img_a.to(dev))
        e2 = model_clip.encode_image(img_b.to(dev))

    e1 = e1 / e1.norm(dim=-1, keepdim=True)
    e2 = e2 / e2.norm(dim=-1, keepdim=True)

    return (e1 @ e2.T).item()


def compute_lpips(model_lpips, img_a, img_b):
    dev = _module_device(model_lpips)
    img_a = img_a.to(dev) * 2 - 1
    img_b = img_b.to(dev) * 2 - 1

    with torch.no_grad():
        d = model_lpips(img_a, img_b)

    return float(d.item())


def hps_score_cpu(img, prompt, version="v2.1"):
    orig_is_avail = torch.cuda.is_available
    torch.cuda.is_available = lambda: False
    try:
        score = hpsv2.score(img, prompt, hps_version=version)
    finally:
        torch.cuda.is_available = orig_is_avail
    return score


def force_pipeline_device_dtype(pipe, device, dtype):
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pipe = pipe.to(device, torch_dtype=dtype)

    # Diffusers Pipeline.to may miss freshly replaced submodules in some notebook states.
    for component_name in ("unet", "vae", "text_encoder", "safety_checker"):
        component = getattr(pipe, component_name, None)
        if isinstance(component, nn.Module):
            component.to(device=device, dtype=dtype)
    return pipe


def move_to_cpu_safely(pipe):
    return force_pipeline_device_dtype(pipe, "cpu", torch.float32)


def prepare_image_for_lpips(pil_img):
    return lpips_transform(pil_img).unsqueeze(0).to(_module_device(lpips_model))


lpips_transform = T.Compose([
    T.Resize((256, 256)),
    T.CenterCrop((256, 256)),
    T.ToTensor(),
])

# %% cell 23
# === Красивая отрисовка двух изображений и метрик под ними ===

def visualize_comparison(
    img_fp,
    img_q,
    lpips_value: float,
    clip_img_sim: float,
    title: str = "FP vs Quantized"
):
    if not isinstance(img_fp, np.ndarray):
        img_fp_np = np.array(img_fp)
    else:
        img_fp_np = img_fp

    if not isinstance(img_q, np.ndarray):
        img_q_np = np.array(img_q)
    else:
        img_q_np = img_q

    diff = np.abs(img_fp_np.astype(float) - img_q_np.astype(float))
    diff_norm = diff / diff.max() if diff.max() > 0 else diff

    plt.figure(figsize=(18, 7))
    plt.subplots_adjust(top=0.80)
    plt.suptitle(
        f"{title}\n"
        f"LPIPS = {lpips_value:.4f}   |   CLIP image-image = {clip_img_sim:.4f}",
        fontsize=14,
        y=0.98
    )

    plt.subplot(1, 3, 1)
    plt.imshow(img_fp_np)
    plt.title("Float UNet")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(img_q_np)
    plt.title("Quantized UNet")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(diff_norm, cmap="inferno")
    plt.title("Difference Heatmap")
    plt.axis("off")

    plt.tight_layout()
    plt.show()

# %% [markdown] cell 24
# <h1>Функции для экспериментов</h1>

# %% cell 25
#=== Функция для привода к скаляру из разных векторов/массивов ===

def to_scalar(x):
    """
    Приводит x к python-float.
    Поддерживает: float/int, torch.Tensor scalar, list/tuple вида [val] или [val,...] (берём среднее).
    """
    if isinstance(x, (float, int)):
        return float(x)
    if torch.is_tensor(x):
        return float(x.detach().item()) if x.numel() == 1 else float(x.detach().mean().item())
    if isinstance(x, (list, tuple)):
        if len(x) == 0:
            return float("nan")
        # часто HPS возвращает [score] или список по батчу
        vals = [to_scalar(v) for v in x]
        return float(sum(vals) / len(vals))
    # на крайний случай
    return float(x)

# %% cell 26
# === VQDM-style выбор слоев, сбор активаций и XTX для activation-aware RVQ ===

def hadamard_coverage_layer(layer: nn.Module, block: int):
    flat_in_dim = layer.weight[0].numel()
    i2 = (flat_in_dim // block) * block
    return i2, i2 / flat_in_dim


def is_vqdm_default_mid_block_layer(
    name: str,
    layer: nn.Module,
    min_features: int = 16,
    include_linear: bool = True,
    include_conv: bool = True,
) -> bool:
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
            res[f"mid_block.{name}"] = layer
    return res


def get_all_mid_block_quant_layers(
    unet: nn.Module,
    min_features: int = 16,
    include_linear: bool = True,
    include_conv: bool = True,
) -> Dict[str, nn.Module]:
    res = {}
    for name, layer in unet.mid_block.named_modules():
        if not name:
            continue
        if include_linear and isinstance(layer, nn.Linear):
            if min(layer.in_features, layer.out_features) >= min_features:
                res[f"mid_block.{name}"] = layer
        elif include_conv and isinstance(layer, nn.Conv2d):
            if min(layer.in_channels, layer.out_channels) >= min_features:
                res[f"mid_block.{name}"] = layer
    return res


def print_mid_block_quant_layers(
    unet: nn.Module,
    min_features: int = 16,
    include_linear: bool = True,
    include_conv: bool = True,
):
    layers = get_mid_block_quant_layers(
        unet,
        min_features=min_features,
        include_linear=include_linear,
        include_conv=include_conv,
    )
    print(f"Found {len(layers)} VQDM-style target layers in mid_block")
    for name, layer in layers.items():
        if isinstance(layer, nn.Linear):
            print(f"{name:60s}  Linear  weight={tuple(layer.weight.shape)}")
        else:
            cols = layer.in_channels * layer.kernel_size[0] * layer.kernel_size[1]
            xtx_mb = cols * cols * 4 / 1024**2
            print(f"{name:60s}  Conv2d  weight={tuple(layer.weight.shape)}  XTX~{xtx_mb:.1f} MiB fp32")
    return layers


def print_all_mid_block_quant_layers(
    unet: nn.Module,
    min_features: int = 16,
    include_linear: bool = True,
    include_conv: bool = True,
):
    layers = get_all_mid_block_quant_layers(
        unet,
        min_features=min_features,
        include_linear=include_linear,
        include_conv=include_conv,
    )
    print(f"Found {len(layers)} all-mid_block target layers")
    for name, layer in layers.items():
        if isinstance(layer, nn.Linear):
            print(f"{name:60s}  Linear  weight={tuple(layer.weight.shape)}")
        else:
            cols = layer.in_channels * layer.kernel_size[0] * layer.kernel_size[1]
            xtx_mb = cols * cols * 4 / 1024**2
            print(f"{name:60s}  Conv2d  weight={tuple(layer.weight.shape)}  XTX~{xtx_mb:.1f} MiB fp32")
    return layers


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
    accum_dtype=torch.float32,
    group_channels=True,
    show_progress=True,
    generator_seed=None,
):
    collect_dtype = globals().get("runtime_dtype", torch.float16) if str(collect_device).startswith("cuda") else torch.float32
    pipe = force_pipeline_device_dtype(pipe, collect_device, collect_dtype)
    xtx_store_device = torch.device(xtx_store_device)
    mid_block = pipe.unet.mid_block

    gen_device = collect_device if str(collect_device).startswith("cuda") else "cpu"
    if generator_seed is not None:
        generator_seed = int(generator_seed)

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
                elif x.ndim != 2:
                    raise ValueError(f"Unexpected Linear input shape for {layer_full_name}: {tuple(x.shape)}")
                x = x.to(xtx_store_device, dtype=torch.float32)
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
                    # VQDM-style order: columns match weight.permute(0, 2, 3, 1).flatten(1)
                    B, _, L = x.shape
                    Cin = module.in_channels
                    Kh, Kw = module.kernel_size
                    x = x.view(B, Cin, Kh, Kw, L).permute(0, 2, 3, 1, 4).contiguous()
                    x = x.view(B, Kh * Kw * Cin, L)

                x = x.permute(1, 0, 2).reshape(x.shape[1], -1).t()
                x = x.to(xtx_store_device, dtype=torch.float32)
                stats[layer_full_name]["xtx"] += x.t().to(accum_dtype) @ x.to(accum_dtype)
                stats[layer_full_name]["n_samples"] += x.shape[0]

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
                    torch.Generator(device=gen_device).manual_seed(generator_seed + start + i)
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
    layer_names=None,
    xtx_batch_size,
    num_inference_steps,
    guidance_scale,
    height=None,
    width=None,
    collect_devices: Sequence[torch.device],
    xtx_store_device="cuda",
    accum_dtype=torch.float32,
    group_channels=True,
    show_progress=True,
    generator_seed=None,
    auto_per_device_batch: bool = True,
    has_conv_targets: bool = False,
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

    log_event(f"Parallel RVQ XTX collection on {[str(d) for d in collect_devices]}")
    replicas = None
    partials = None
    try:
        replicas = replicate_text2image_pipeline_for_devices(
            force_pipeline_device_dtype(pipe, str(collect_devices[0]), runtime_dtype),
            collect_devices,
            detach=True,
            reuse_first_replica=True,
        )
        device_weights = [_device_vram_weight(dev) for dev in collect_devices]
        chunks = _even_prompt_chunks(calibration_prompts, len(collect_devices), weights=device_weights)
        batch_cap = max(1, int(xtx_batch_size))

        def worker(replica, device, start, chunk_prompts):
            if auto_per_device_batch:
                local_auto = resolve_xtx_batch_size(
                    len(chunk_prompts),
                    requested_batch=0,
                    has_conv_targets=has_conv_targets,
                    device=device,
                )
                local_batch = max(1, min(int(local_auto), batch_cap, len(chunk_prompts)))
            else:
                local_batch = max(1, min(batch_cap, len(chunk_prompts)))
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
                generator_seed=None if generator_seed is None else int(generator_seed) + int(start),
            )

        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=len(chunks)) as executor:
            futures = [
                executor.submit(worker, replicas[i], collect_devices[i], start, chunk_prompts)
                for i, (start, chunk_prompts) in enumerate(chunks)
            ]
            partials = [future.result() for future in futures]

        merged = {}
        names = list(layer_names) if layer_names is not None else list(partials[0].keys())
        target_device = torch.device(xtx_store_device)
        for full_name in names:
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
        _clear_cuda_after_oom()


@torch.no_grad()
def collect_midblock_xtx_streaming_autobatch(
    *,
    pipe,
    calibration_prompts,
    layer_names=None,
    xtx_batch_size=0,
    num_inference_steps=50,
    guidance_scale=5.0,
    height=None,
    width=None,
    collect_device="cuda",
    xtx_store_device="cpu",
    accum_dtype=torch.float32,
    group_channels=True,
    show_progress=True,
    generator_seed=None,
):
    requested = int(xtx_batch_size or 0)
    has_conv_targets = _rvq_has_conv_targets(pipe, layer_names)
    collect_devices = _rvq_xtx_devices()
    if len(collect_devices) <= 1:
        collect_devices = [torch.device(collect_device)]
    start_batch = resolve_xtx_batch_size(
        len(calibration_prompts),
        requested_batch=requested,
        has_conv_targets=has_conv_targets,
        device=collect_devices[0] if str(collect_devices[0]).startswith("cuda") else None,
    )
    if requested <= 0 and len(collect_devices) > 1:
        device_weights = [_device_vram_weight(dev) for dev in collect_devices]
        chunks = _even_prompt_chunks(calibration_prompts, len(collect_devices), weights=device_weights)
        per_device_batches = [
            resolve_xtx_batch_size(
                len(chunk_prompts),
                requested_batch=0,
                has_conv_targets=has_conv_targets,
                device=dev,
            )
            for dev, (_, chunk_prompts) in zip(collect_devices, chunks)
        ]
        if per_device_batches:
            start_batch = max(start_batch, max(per_device_batches))
    batch = start_batch

    while True:
        try:
            log_event(f"Collect XTX with xtx_batch_size_cap={batch}, devices={[str(d) for d in collect_devices]}")
            if len(collect_devices) > 1:
                return collect_midblock_xtx_streaming_parallel(
                    pipe=pipe,
                    calibration_prompts=calibration_prompts,
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
                    auto_per_device_batch=(requested <= 0),
                    has_conv_targets=has_conv_targets,
                )
            return collect_midblock_xtx_streaming(
                pipe=pipe,
                calibration_prompts=calibration_prompts,
                layer_names=layer_names,
                xtx_batch_size=batch,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                height=height,
                width=width,
                collect_device=collect_device,
                xtx_store_device=xtx_store_device,
                accum_dtype=accum_dtype,
                group_channels=group_channels,
                show_progress=show_progress,
                generator_seed=generator_seed,
            )
        except RuntimeError as exc:
            if _is_cuda_oom(exc) and batch > 1:
                old_batch = batch
                batch = max(1, batch // 2)
                _clear_cuda_after_oom()
                log_event(f"CUDA OOM during XTX collection; reducing xtx_batch_size {old_batch} -> {batch}")
                continue
            raise


@torch.no_grad()
def transform_xtx_hadamard_blocks(
    xtx: torch.Tensor,
    block: int,
    seed: int,
    with_signs: bool = True,
):
    H = xtx.detach().clone().float()
    hadamard_right_blocks(H, block=block, seed=seed, with_signs=with_signs)
    HT = H.t().contiguous()
    hadamard_right_blocks(HT, block=block, seed=seed, with_signs=with_signs)
    return HT.t().contiguous()


@torch.no_grad()
def xtx_block_energy_stats(xtx: torch.Tensor, meta: VectorizeMeta) -> Dict[str, float]:
    padded_dim = meta.n_chunks * meta.d
    H = xtx.detach()
    if H.shape[0] != meta.last_dim or H.shape[1] != meta.last_dim:
        raise ValueError(f"XTX shape {tuple(H.shape)} does not match last_dim={meta.last_dim}")
    if padded_dim > meta.last_dim:
        H = F.pad(H, (0, padded_dim - meta.last_dim, 0, padded_dim - meta.last_dim))

    H4 = H.reshape(meta.n_chunks, meta.d, meta.n_chunks, meta.d)
    idx = torch.arange(meta.n_chunks, device=H.device)
    diag_blocks = H4[idx, :, idx, :]
    total = H.square().sum().clamp_min(1e-30)
    diag = diag_blocks.square().sum()
    offdiag = (total - diag).clamp_min(0.0)
    return {
        "total": float(total.item()),
        "diag": float(diag.item()),
        "offdiag": float(offdiag.item()),
        "offdiag_ratio": float((offdiag / total).item()),
        "diag_ratio": float((diag / total).item()),
        "offdiag_to_diag": float((offdiag / diag.clamp_min(1e-30)).item()),
    }


def _log_xtx_block_energy(
    *,
    layer_salt: int,
    meta: VectorizeMeta,
    cfg: VQConfig,
    before: Dict[str, float],
    after: Dict[str, float],
):
    entry = {
        "layer_salt": int(layer_salt),
        "d": int(meta.d),
        "n_chunks": int(meta.n_chunks),
        "last_dim": int(meta.last_dim),
        "hadamard": bool(cfg.use_hadamard),
        "hadamard_block": int(cfg.hadamard_block),
        "hadamard_with_signs": bool(cfg.hadamard_with_signs),
        "before": dict(before),
        "after": dict(after),
    }
    globals().setdefault("rvq_xtx_block_energy_logs", []).append(entry)
    log_event(
        "[RVQ XTX block energy] "
        f"layer_salt={layer_salt}, d={meta.d}, chunks={meta.n_chunks}, "
        f"hadamard={cfg.use_hadamard}, block={cfg.hadamard_block}, signs={cfg.hadamard_with_signs} | "
        f"offdiag_ratio before={before['offdiag_ratio']:.6f}, after={after['offdiag_ratio']:.6f}, "
        f"delta={after['offdiag_ratio'] - before['offdiag_ratio']:+.6f}"
    )


@torch.no_grad()
def make_vector_hessian_blocks(
    xtx: Optional[torch.Tensor],
    meta: VectorizeMeta,
    cfg: VQConfig,
    layer_salt: int,
    device,
):
    if xtx is None or not cfg.use_activation_aware:
        return None, None, None

    H = xtx.detach().to(device=device, dtype=torch.float32)
    H = 0.5 * (H + H.t())
    if H.shape[0] != meta.last_dim:
        raise ValueError(f"XTX shape {tuple(H.shape)} does not match last_dim={meta.last_dim}")

    before_energy = xtx_block_energy_stats(H, meta) if cfg.log_xtx_block_energy else None

    if cfg.use_hadamard:
        H = transform_xtx_hadamard_blocks(
            H,
            block=cfg.hadamard_block,
            seed=cfg.hadamard_seed + cfg.seed + layer_salt,
            with_signs=cfg.hadamard_with_signs,
        )

    after_energy = xtx_block_energy_stats(H, meta) if cfg.log_xtx_block_energy else None
    if before_energy is not None and after_energy is not None:
        _log_xtx_block_energy(
            layer_salt=layer_salt,
            meta=meta,
            cfg=cfg,
            before=before_energy,
            after=after_energy,
        )

    padded_dim = meta.n_chunks * meta.d
    if H.shape[0] != meta.last_dim:
        raise ValueError(f"XTX shape {tuple(H.shape)} не совпадает с last_dim={meta.last_dim}")

    if padded_dim > meta.last_dim:
        H = F.pad(H, (0, padded_dim - meta.last_dim, 0, padded_dim - meta.last_dim))
    H_full = H.contiguous()

    H4 = H.view(meta.n_chunks, meta.d, meta.n_chunks, meta.d)
    idx = torch.arange(meta.n_chunks, device=device)
    H_blocks = H4[idx, :, idx, :].contiguous()
    H_blocks = 0.5 * (H_blocks + H_blocks.transpose(-1, -2))

    eye = torch.eye(meta.d, device=device, dtype=torch.float32)
    diag_mean = H_blocks.diagonal(dim1=-2, dim2=-1).mean(dim=-1).clamp_min(1e-12)
    H_blocks = H_blocks + (cfg.xtx_damping * diag_mean)[:, None, None] * eye

    if cfg.xtx_normalize_blocks:
        H_blocks = H_blocks / diag_mean[:, None, None]

    rows = meta.prefix_shape[0]
    block_ids = torch.arange(meta.n_chunks, device=device).repeat(rows)
    return H_blocks, block_ids, H_full


@torch.no_grad()
def quantize_mid_block_rvq(
    pipe_fp: StableDiffusionPipeline,
    calibration_prompts: Sequence[str],
    cfg: VQConfig,
    layer_names: Optional[Sequence[str]] = None,
    xtx_stats: Optional[Dict[str, Any]] = None,
    xtx_batch_size: int = 4,
    num_inference_steps: int = 30,
    guidance_scale: float = 7.5,
    height: Optional[int] = None,
    width: Optional[int] = None,
    min_features: int = 16,
    include_linear: bool = True,
    include_conv: bool = True,
    collect_device: str = "cuda",
    xtx_store_device: str = "cpu",
    final_device: str = "cpu",
    final_dtype: torch.dtype = torch.float32,
    verbose: bool = True,
) -> nn.Module:
    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)

    all_layers = get_mid_block_quant_layers(
        pipe_fp.unet,
        min_features=min_features,
        include_linear=include_linear,
        include_conv=include_conv,
    )
    quant_order = list(all_layers.keys()) if layer_names is None else list(layer_names)
    target_names = set(quant_order)

    if cfg.use_activation_aware and xtx_stats is None:
        if verbose:
            print(f"[RVQ] Collect XTX for {len(quant_order)} mid_block layers")
        xtx_stats = collect_midblock_xtx_streaming_autobatch(
            pipe=pipe_fp,
            calibration_prompts=calibration_prompts,
            layer_names=quant_order,
            xtx_batch_size=xtx_batch_size,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            height=height,
            width=width,
            collect_device=collect_device,
            xtx_store_device=xtx_store_device,
            accum_dtype=torch.float32,
            group_channels=cfg.group_channels,
            show_progress=verbose,
            generator_seed=globals().get("rvq_xtx_seed", 1234),
        )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return quantize_unet_all_weights(
        pipe_fp.unet.mid_block,
        cfg=cfg,
        xtx_stats=xtx_stats,
        target_layer_names=target_names,
        prefix="mid_block",
        final_device=final_device,
        final_dtype=final_dtype,
    )

def compute_rvq_compression_for_layers(
    layers: Dict[str, nn.Module],
    cfg: VQConfig,
    fp_bits: int = 16,
    codebook_bits: int = 16,
    bias_bits: Optional[int] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    if bias_bits is None:
        bias_bits = fp_bits

    d = int(cfg.d)
    stage_ks = _cfg_stage_ks(cfg)
    stage_bits = _cfg_stage_bits(cfg)
    M = len(stage_ks)
    K = max(stage_ks)
    bits = max(stage_bits)
    index_bits_per_vector = sum(stage_bits)
    codebook_entries = sum(stage_ks)

    total_weight_params = 0
    total_weight_fp_bits = 0
    total_bias_params = 0
    total_bias_bits = 0
    total_vectors = 0
    total_pad_params = 0
    total_index_bits = 0
    total_codebook_bits = 0
    per_layer = []

    for name, layer in layers.items():
        stats = _weight_stats(layer, d=d)
        weight_params = stats["orig_params"]
        n_vectors = stats["n_vectors"]
        pad_params = stats["pad_params"]
        bias_params = layer.bias.numel() if layer.bias is not None else 0

        index_bits = n_vectors * index_bits_per_vector
        cb_bits = codebook_entries * d * codebook_bits

        total_weight_params += weight_params
        total_weight_fp_bits += weight_params * fp_bits
        total_bias_params += bias_params
        total_bias_bits += bias_params * bias_bits
        total_vectors += n_vectors
        total_pad_params += pad_params
        total_index_bits += index_bits
        total_codebook_bits += cb_bits

        per_layer.append({
            "name": name,
            "layer_type": type(layer).__name__,
            "weight_shape": tuple(layer.weight.shape),
            "weight_params": weight_params,
            "bias_params": bias_params,
            "n_vectors": n_vectors,
            "pad_params": pad_params,
            "index_bits": index_bits,
            "codebook_bits": cb_bits,
            "bpp_weight_only": (index_bits + cb_bits) / weight_params,
        })

    quant_bits = total_index_bits + total_codebook_bits
    result = {
        "config": {"bits": bits, "K": K, "stage_Ks": list(stage_ks), "stage_bits": list(stage_bits), "d": d, "rvq_M": M},
        "totals": {
            "target_layers": len(layers),
            "target_weight_params": total_weight_params,
            "target_weight_fp_bits": total_weight_fp_bits,
            "target_bias_params": total_bias_params,
            "target_vectors": total_vectors,
            "target_pad_params": total_pad_params,
            "index_bits": total_index_bits,
            "codebook_bits": total_codebook_bits,
            "quant_weight_bits": quant_bits,
            "quant_weight_bpp": quant_bits / total_weight_params if total_weight_params else 0.0,
            "quant_weight_compression": total_weight_fp_bits / quant_bits if quant_bits else float("inf"),
        },
        "per_layer": per_layer,
    }

    if verbose:
        t = result["totals"]
        print("=== RVQ compression stats (target VQDM-style mid_block layers) ===")
        print(f"Target layers                    : {t['target_layers']:,}")
        print(f"Target weight params             : {t['target_weight_params']:,}")
        print(f"Bias params in target layers     : {t['target_bias_params']:,}")
        print(f"Total RVQ vectors                : {t['target_vectors']:,}")
        print(f"Total padded params              : {t['target_pad_params']:,}")
        print()
        print(f"Index bits                       : {t['index_bits']:,}")
        print(f"Codebook bits                    : {t['codebook_bits']:,}")
        print(f"Target weight FP bits            : {t['target_weight_fp_bits']:,}")
        print()
        print(f"Target weight bpp                : {t['quant_weight_bpp']:.6f}")
        print(f"Target weight compression        : {t['quant_weight_compression']:.4f}x")

    return result

# %% cell 27
# === Image generation and metric evaluation ===

@torch.no_grad()
def generate_image(
    pipe,
    prompt,
    seed,
    num_inference_steps=50,
    guidance_scale=5.0,
    height=None,
    width=None,
    device=None,
    dtype=None,
    keep_on_device=None,
):
    device = device or globals().get("generation_device", runtime_device)
    dtype = dtype or globals().get("generation_dtype", runtime_dtype)
    keep_on_device = globals().get("generation_keep_on_device", True) if keep_on_device is None else keep_on_device

    gen_device = device if str(device).startswith("cuda") else "cpu"
    gen = torch.Generator(device=gen_device).manual_seed(seed)
    pipe = force_pipeline_device_dtype(pipe, device, dtype)
    pipe.set_progress_bar_config(disable=True)
    img = pipe(
        prompt,
        generator=gen,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        height=height,
        width=width,
    ).images[0]

    if not keep_on_device:
        move_to_cpu_safely(pipe)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return img


@torch.no_grad()
def generate_images_batched(
    pipe,
    prompts_batch,
    seeds_batch,
    num_inference_steps=50,
    guidance_scale=5.0,
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
    if len(prompts_batch) == 1:
        return [generate_image(
            pipe,
            prompts_batch[0],
            seeds_batch[0],
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            height=height,
            width=width,
            device=device,
            dtype=dtype,
            keep_on_device=keep_on_device,
        )]

    device = device or globals().get("generation_device", runtime_device)
    dtype = dtype or globals().get("generation_dtype", runtime_dtype)
    keep_on_device = globals().get("generation_keep_on_device", True) if keep_on_device is None else keep_on_device
    gen_device = device if str(device).startswith("cuda") else "cpu"

    try:
        gens = [torch.Generator(device=gen_device).manual_seed(s) for s in seeds_batch]
        pipe = force_pipeline_device_dtype(pipe, device, dtype)
        pipe.set_progress_bar_config(disable=True)
        images = pipe(
            prompts_batch,
            generator=gens,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            height=height,
            width=width,
        ).images
        if not keep_on_device:
            move_to_cpu_safely(pipe)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        return list(images)
    except RuntimeError as exc:
        if _is_cuda_oom(exc) and len(prompts_batch) > 1:
            _clear_cuda_after_oom()
            mid = len(prompts_batch) // 2
            log_event(
                f"CUDA OOM in batched generation; splitting batch {len(prompts_batch)} -> {mid}+{len(prompts_batch)-mid}"
            )
            left = generate_images_batched(
                pipe,
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
            right = generate_images_batched(
                pipe,
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
            if not keep_on_device:
                move_to_cpu_safely(pipe)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            return left + right
        raise


def optimize_pipeline_for_inference(pipe_obj, label: str = "pipeline"):
    """Apply low-risk inference optimizations for SD pipelines."""
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


@torch.no_grad()
def prepare_images_for_clip(imgs):
    return torch.stack([clip_preprocess(img) for img in imgs], dim=0).to(_module_device(clip_model))


@torch.no_grad()
def prepare_images_for_lpips(imgs):
    return torch.stack([lpips_transform(img) for img in imgs], dim=0).to(_module_device(lpips_model))


@torch.no_grad()
def compute_pixel_metrics_batch(img_fp_pils, img_test_pils):
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
def compute_metrics_batch_against_fp(img_fp_pils, img_test_pils, prompts_batch):
    dev_clip = _module_device(clip_model)
    prompts_batch = list(prompts_batch)

    img_fp_clip = prepare_images_for_clip(img_fp_pils)
    img_t_clip = prepare_images_for_clip(img_test_pils)
    txt = clip_tokenizer(prompts_batch).to(dev_clip)

    txt_emb = clip_model.encode_text(txt)
    fp_emb = clip_model.encode_image(img_fp_clip.to(dev_clip))
    test_emb = clip_model.encode_image(img_t_clip.to(dev_clip))

    txt_emb = txt_emb / txt_emb.norm(dim=-1, keepdim=True)
    fp_emb = fp_emb / fp_emb.norm(dim=-1, keepdim=True)
    test_emb = test_emb / test_emb.norm(dim=-1, keepdim=True)

    clip_text_fp = (txt_emb * fp_emb).sum(dim=-1).detach().float().cpu().tolist()
    clip_text_test = (txt_emb * test_emb).sum(dim=-1).detach().float().cpu().tolist()
    clip_img_sim = (fp_emb * test_emb).sum(dim=-1).detach().float().cpu().tolist()

    img_fp_lp = prepare_images_for_lpips(img_fp_pils).to(_module_device(lpips_model)) * 2 - 1
    img_t_lp = prepare_images_for_lpips(img_test_pils).to(_module_device(lpips_model)) * 2 - 1
    lp = lpips_model(img_fp_lp, img_t_lp).view(-1).detach().float().cpu().tolist()
    pix = compute_pixel_metrics_batch(img_fp_pils, img_test_pils)

    return [
        {
            "clip_text_fp": float(clip_text_fp[i]),
            "clip_text_test": float(clip_text_test[i]),
            "delta_clip_text": float(clip_text_test[i] - clip_text_fp[i]),
            "clip_img_sim": float(clip_img_sim[i]),
            "lpips": float(lp[i]),
            **pix[i],
        }
        for i in range(len(prompts_batch))
    ]


@torch.no_grad()
def evaluate_against_fp(
    img_fp_pil,
    pipe_test,
    prompt,
    clip_model,
    clip_tokenizer,
    lpips_model,
    seed=5,
    num_inference_steps=50,
    guidance_scale=5.0,
    height=None,
    width=None,
    image_pbar=None,
    keep_on_device=None,
):
    img_test_pil = generate_image(
        pipe_test,
        prompt,
        seed,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        height=height,
        width=width,
        keep_on_device=keep_on_device,
    )
    if image_pbar is not None:
        image_pbar.update(1)

    img_fp_clip = prepare_image_for_clip(img_fp_pil)
    img_t_clip = prepare_image_for_clip(img_test_pil)

    clip_text_fp = compute_clip_text_image_score(clip_model, clip_tokenizer, prompt, img_fp_clip)
    clip_text_t = compute_clip_text_image_score(clip_model, clip_tokenizer, prompt, img_t_clip)
    clip_img_sim = compute_clip_image_image_score(clip_model, img_fp_clip, img_t_clip)

    img_fp_lp = prepare_image_for_lpips(img_fp_pil)
    img_t_lp = prepare_image_for_lpips(img_test_pil)
    lpips_val = compute_lpips(lpips_model, img_fp_lp, img_t_lp)
    pix = compute_pixel_metrics_batch([img_fp_pil], [img_test_pil])[0]

    return {
        "clip_text_fp": float(clip_text_fp),
        "clip_text_test": float(clip_text_t),
        "delta_clip_text": float(clip_text_t - clip_text_fp),
        "clip_img_sim": float(clip_img_sim),
        "lpips": float(lpips_val),
        **pix,
        "img_fp": img_fp_pil,
        "img_test": img_test_pil,
    }

# %% cell 28
# === UNet forward error ===

@torch.no_grad()
def unet_rel_error(pipe_ref, pipe_test, prompt: str, sample, t_rand):
    device = sample.device
    dtype = runtime_dtype if device.type == "cuda" else torch.float32

    pipe_ref.unet.to(device, dtype=dtype)
    pipe_test.unet.to(device, dtype=dtype)
    pipe_ref.text_encoder.to(device, dtype=dtype)

    text_inputs = pipe_ref.tokenizer(
        [prompt],
        padding="max_length",
        max_length=pipe_ref.tokenizer.model_max_length,
        return_tensors="pt",
    )

    input_ids = text_inputs.input_ids.to(device)
    encoder_hidden_states = pipe_ref.text_encoder(input_ids)[0].to(dtype)

    out_fp = pipe_ref.unet(sample, t_rand, encoder_hidden_states=encoder_hidden_states).sample
    out_q = pipe_test.unet(sample, t_rand, encoder_hidden_states=encoder_hidden_states).sample

    diff = out_fp.float() - out_q.float()
    rel = diff.norm().item() / (out_fp.float().norm().item() + 1e-12)
    mx = diff.abs().max().item()

    if not globals().get("generation_keep_on_device", True):
        move_to_cpu_safely(pipe_ref)
        move_to_cpu_safely(pipe_test)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return rel, mx


RVQ_FORWARD_METRIC_PROMPTS = _env_int("RVQ_FORWARD_METRIC_PROMPTS", 2)
RVQ_FORWARD_METRIC_TIMESTEPS = _env_int("RVQ_FORWARD_METRIC_TIMESTEPS", 3)
RVQ_FORWARD_METRIC_SEED = _env_int("RVQ_FORWARD_METRIC_SEED", 20260515)


def _rvq_forward_metric_timesteps(n_steps: int) -> List[int]:
    n_steps = max(1, int(n_steps))
    scheduler_config = getattr(getattr(pipe, "scheduler", None), "config", None)
    total = int(getattr(scheduler_config, "num_train_timesteps", 1000))
    hi = max(1, total - max(1, total // 10))
    lo = max(1, total // 10)
    if n_steps == 1:
        return [int((hi + lo) // 2)]
    return [int(round(x)) for x in np.linspace(hi, lo, n_steps)]


def _rvq_forward_metric_prompts(prompts: Sequence[str]) -> List[str]:
    prompts_list = [str(p) for p in prompts]
    limit = max(1, int(RVQ_FORWARD_METRIC_PROMPTS))
    return prompts_list[:limit] if prompts_list else [""]


def _rvq_compare_tensors(prefix: str, fp: torch.Tensor, q: torch.Tensor) -> Dict[str, float]:
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
def _rvq_make_forward_probe_inputs(pipe_obj, prompts: Sequence[str]):
    device = torch.device(rvq_eval_device if torch.cuda.is_available() else "cpu")
    dtype = rvq_final_dtype if device.type == "cuda" else torch.float32
    probe_prompts = _rvq_forward_metric_prompts(prompts)

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
    generator = torch.Generator(device=device).manual_seed(int(RVQ_FORWARD_METRIC_SEED))
    sample = torch.randn(
        len(probe_prompts),
        int(pipe_obj.unet.config.in_channels),
        height,
        width,
        generator=generator,
        device=device,
        dtype=dtype,
    )
    timesteps = _rvq_forward_metric_timesteps(RVQ_FORWARD_METRIC_TIMESTEPS)
    return sample, timesteps, encoder_hidden_states


@torch.no_grad()
def _rvq_run_unet_forward_probe(pipe_obj, sample: torch.Tensor, timesteps: Sequence[int], encoder_hidden_states: torch.Tensor):
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
def compute_forward_drift_between_pipes(pipe_fp, pipe_test, prompts: Sequence[str], eval_label: str = "RVQ") -> Dict[str, float]:
    if int(RVQ_FORWARD_METRIC_PROMPTS) <= 0 or int(RVQ_FORWARD_METRIC_TIMESTEPS) <= 0:
        return {}
    try:
        sample, timesteps, encoder_hidden_states = _rvq_make_forward_probe_inputs(pipe_fp, prompts)
        pipe_test.unet.to(sample.device, dtype=sample.dtype)
        fp_out, fp_mid = _rvq_run_unet_forward_probe(pipe_fp, sample, timesteps, encoder_hidden_states)
        q_out, q_mid = _rvq_run_unet_forward_probe(pipe_test, sample, timesteps, encoder_hidden_states)
        metrics = {}
        metrics.update(_rvq_compare_tensors("unet_forward", fp_out, q_out))
        metrics.update(_rvq_compare_tensors("midblock_forward", fp_mid, q_mid))
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
def compute_forward_drift_with_midblock_patch(
    pipe_obj,
    mid_block_q: nn.Module,
    prompts: Sequence[str],
    eval_label: str = "RVQ",
) -> Dict[str, float]:
    if int(RVQ_FORWARD_METRIC_PROMPTS) <= 0 or int(RVQ_FORWARD_METRIC_TIMESTEPS) <= 0:
        return {}
    fp_mid_block = pipe_obj.unet.mid_block
    fp_was_training = pipe_obj.unet.training
    try:
        sample, timesteps, encoder_hidden_states = _rvq_make_forward_probe_inputs(pipe_obj, prompts)
        fp_out, fp_mid = _rvq_run_unet_forward_probe(pipe_obj, sample, timesteps, encoder_hidden_states)
        pipe_obj.unet.mid_block = mid_block_q.to(sample.device, dtype=sample.dtype)
        pipe_obj.unet.eval()
        q_out, q_mid = _rvq_run_unet_forward_probe(pipe_obj, sample, timesteps, encoder_hidden_states)
        metrics = {}
        metrics.update(_rvq_compare_tensors("unet_forward", fp_out, q_out))
        metrics.update(_rvq_compare_tensors("midblock_forward", fp_mid, q_mid))
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

# %% cell 29
# === Metrics over prompts and seeds ===

def rvq_eval_pairs(prompts, seeds):
    return [(prompt, int(s)) for prompt in prompts for s in seeds]


def rvq_fp_cache_key(prompt, seed, num_inference_steps=None, guidance_scale=None, height=None, width=None):
    return (
        prompt,
        int(seed),
        int(num_inference_steps if num_inference_steps is not None else globals().get("num_inference_steps", 30)),
        float(guidance_scale if guidance_scale is not None else globals().get("guidance_scale", 7.5)),
        height,
        width,
    )


_RVQ_GEN_BATCH_LOG_CACHE = {}


def resolve_generation_batch_size(total_items: Optional[int] = None, label: str = "generation") -> int:
    """Resolve an aggressive but safe Stable Diffusion generation batch size."""
    total_items_int = int(total_items) if total_items is not None else None

    if RVQ_GEN_BATCH_ENV > 0:
        batch = int(RVQ_GEN_BATCH_ENV)
        reason = "env"
    elif not torch.cuda.is_available() or not str(globals().get("generation_device", runtime_device)).startswith("cuda"):
        batch = 1
        reason = "cpu"
    else:
        info = _cuda_mem_info() or {"free_gib": 0.0, "total_gib": 0.0}
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
        auto_cap = int(RVQ_GEN_BATCH_MAX) if int(RVQ_GEN_BATCH_MAX) > 0 else (64 if total_gib >= 70.0 else 32)
        batch = min(batch, auto_cap)
        reason = f"auto free={free_gib:.1f}GiB total={total_gib:.1f}GiB cap={auto_cap}"

    if total_items_int is not None and total_items_int > 0:
        batch = min(batch, total_items_int)
    batch = max(1, int(batch))

    cache_key = (label, total_items_int, batch, reason)
    if _RVQ_GEN_BATCH_LOG_CACHE.get(label) != cache_key:
        _RVQ_GEN_BATCH_LOG_CACHE[label] = cache_key
        log_event(f"Resolved generation batch | {label}: batch={batch}, total_items={total_items_int}, reason={reason}")
    return batch


@torch.no_grad()
def precompute_fp_image_cache(
    pipe_fp,
    prompts,
    seeds,
    cache=None,
    image_pbar=None,
    num_inference_steps=None,
    guidance_scale=None,
    height=None,
    width=None,
):
    cache = {} if cache is None else cache
    steps = int(num_inference_steps if num_inference_steps is not None else globals().get("num_inference_steps", 30))
    scale = float(guidance_scale if guidance_scale is not None else globals().get("guidance_scale", 7.5))
    keep_after_batch = globals().get("generation_keep_on_device", True)

    try:
        pending = []
        for prompt, s in rvq_eval_pairs(prompts, seeds):
            key = rvq_fp_cache_key(prompt, s, steps, scale, height, width)
            if key not in cache:
                pending.append((prompt, int(s), key))

        gen_batch = resolve_generation_batch_size(len(pending), label="FP cache")
        for b0 in range(0, len(pending), gen_batch):
            chunk = pending[b0:b0 + gen_batch]
            if not chunk:
                continue
            chunk_prompts = [x[0] for x in chunk]
            chunk_seeds = [x[1] for x in chunk]
            if image_pbar is not None:
                image_pbar.set_postfix_str(f"FP batch={len(chunk)}, seed={chunk_seeds[0]}")
            imgs = generate_images_batched(
                pipe_fp,
                chunk_prompts,
                chunk_seeds,
                num_inference_steps=steps,
                guidance_scale=scale,
                height=height,
                width=width,
                keep_on_device=True,  # keep FP pipeline on GPU for the whole cache batch
            )
            for (_, _, key), img in zip(chunk, imgs):
                cache[key] = img
            if image_pbar is not None:
                image_pbar.update(len(chunk))
    finally:
        if not keep_after_batch:
            move_to_cpu_safely(pipe_fp)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    return cache


def count_missing_fp_images(cache, prompts, seeds, num_inference_steps=None, guidance_scale=None, height=None, width=None):
    cache = {} if cache is None else cache
    steps = int(num_inference_steps if num_inference_steps is not None else globals().get("num_inference_steps", 30))
    scale = float(guidance_scale if guidance_scale is not None else globals().get("guidance_scale", 7.5))
    return sum(
        rvq_fp_cache_key(prompt, s, steps, scale, height, width) not in cache
        for prompt, s in rvq_eval_pairs(prompts, seeds)
    )


@torch.no_grad()
def eval_prompts_against_fp(
    pipe_fp,
    pipe_test,
    prompts,
    seeds,
    fp_cache=None,
    image_pbar=None,
    eval_label="RVQ",
    forward_drift_metrics: Optional[Dict[str, float]] = None,
    skip_forward_drift: bool = False,
):
    rows = []
    pairs = rvq_eval_pairs(prompts, seeds)
    fp_cache = {} if fp_cache is None else fp_cache
    keep_after_batch = globals().get("generation_keep_on_device", True)
    forward_drift_metrics = dict(forward_drift_metrics or {})

    precompute_fp_image_cache(
        pipe_fp,
        prompts,
        seeds,
        cache=fp_cache,
        image_pbar=image_pbar,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
    )

    try:
        gen_batch = resolve_generation_batch_size(len(pairs), label=eval_label)
        for b0 in range(0, len(pairs), gen_batch):
            chunk = pairs[b0:b0 + gen_batch]
            chunk_prompts = [p for p, _ in chunk]
            chunk_seeds = [int(s) for _, s in chunk]
            if image_pbar is not None:
                image_pbar.set_postfix_str(f"{eval_label} batch={len(chunk)}, seed={chunk_seeds[0]}")

            imgs_fp = [fp_cache[rvq_fp_cache_key(prompt, s, num_inference_steps, guidance_scale)] for prompt, s in chunk]
            t_gen0 = time.perf_counter()
            imgs_test = generate_images_batched(
                pipe_test,
                chunk_prompts,
                chunk_seeds,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                keep_on_device=True,  # keep quantized pipeline on GPU for the whole eval batch
            )
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t_gen = time.perf_counter() - t_gen0
            if image_pbar is not None:
                image_pbar.update(len(chunk))
                image_pbar.set_postfix_str(f"{eval_label} metrics batch={len(chunk)}")

            t_met0 = time.perf_counter()
            metrics = compute_metrics_batch_against_fp(imgs_fp, imgs_test, chunk_prompts)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t_met = time.perf_counter() - t_met0
            log_event(
                f"Eval batch | {eval_label} | n={len(chunk)}, gen={t_gen:.2f}s, metrics={t_met:.2f}s, "
                f"gen_batch={gen_batch}"
            )

            for (prompt, s), img_fp, img_test, res in zip(chunk, imgs_fp, imgs_test, metrics):
                if globals().get("rvq_compute_hps", False):
                    hps_fp = to_scalar(hps_score_cpu(img_fp, prompt))
                    hps_test = to_scalar(hps_score_cpu(img_test, prompt))
                else:
                    hps_fp = float("nan")
                    hps_test = float("nan")

                rows.append({
                    "prompt": prompt,
                    "seed": int(s),
                    "lpips": float(res["lpips"]),
                    "clip_ti_fp": float(res["clip_text_fp"]),
                    "clip_ti": float(res["clip_text_test"]),
                    "delta_clip_ti": float(res["delta_clip_text"]),
                    "clip_ii": float(res["clip_img_sim"]),
                    "pixel_mse": float(res["pixel_mse"]),
                    "pixel_mae": float(res["pixel_mae"]),
                    "psnr": float(res["psnr"]),
                    "ssim": float(res["ssim"]),
                    "hps": float(hps_test),
                    "hps_fp": float(hps_fp),
                })

        if not skip_forward_drift and not forward_drift_metrics:
            forward_drift_metrics = compute_forward_drift_between_pipes(pipe_fp, pipe_test, prompts, eval_label=eval_label)
    finally:
        if not keep_after_batch:
            move_to_cpu_safely(pipe_test)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def mean(xs):
        return sum(xs) / len(xs)

    def mean_finite(xs):
        vals = [x for x in xs if np.isfinite(x)]
        return sum(vals) / len(vals) if vals else float("nan")

    agg = {
        "lpips": mean([r["lpips"] for r in rows]),
        "clip_ti_fp": mean([r["clip_ti_fp"] for r in rows]),
        "clip_ti": mean([r["clip_ti"] for r in rows]),
        "delta_clip_ti": mean([r["delta_clip_ti"] for r in rows]),
        "clip_ii": mean([r["clip_ii"] for r in rows]),
        "pixel_mse": mean([r["pixel_mse"] for r in rows]),
        "pixel_mae": mean([r["pixel_mae"] for r in rows]),
        "psnr": mean([r["psnr"] for r in rows]),
        "ssim": mean([r["ssim"] for r in rows]),
        "hps": mean_finite([r["hps"] for r in rows]),
        "hps_fp": mean_finite([r["hps_fp"] for r in rows]),
        "n": len(rows),
    }
    agg.update(forward_drift_metrics)
    return agg, rows

# %% [markdown] cell 30
# <h1>Запуск самого квантования и использование всех раннее сделанных функций</h1>

# %% cell 31
# # === Всего Unet'а квантизация ===

# unet = pipe.unet
# unet_q_all = quantize_unet_all_weights(pipe.unet, VQConfig(bits=4, d=16, max_iter=10, on_cpu=False))

# H = W = getattr(unet_.config, "sample_size", 64)
# B = 1

# sample = torch.randn(
#     B, unet.config.in_channels, H, W,
#     device="cuda",
#     dtype=torch.float16
# )

# t = torch.randint(0, 1000, (B,), device="cuda", dtype=torch.long)

# prompt = "a photo of a cute cat, high quality, 4k"
# text_inputs = pipe.tokenizer(
#     [prompt],
#     padding="max_length",
#     max_length=pipe.tokenizer.model_max_length,
#     return_tensors="pt"
# )
# encoder_hidden_states = pipe.text_encoder(
#     text_inputs.input_ids.to("cuda")
# )[0].to(torch.float16)

# with torch.no_grad():
#     out_fp = unet(
#         sample,
#         t,
#         encoder_hidden_states=encoder_hidden_states
#     ).sample

#     out_q_all = unet_q_all(
#         sample,
#         t,
#         encoder_hidden_states=encoder_hidden_states
#     ).sample

# diff = out_fp.float() - out_q_all.float()
# num = diff.pow(2).sum().sqrt()
# den = out_fp.float().pow(2).sum().sqrt()

# rel_mse_unet_all = (num / den).item()
# max_diff_unet_all = diff.abs().max().item()

# print("rel MSE всего UNet (weights-only quant):", rel_mse_unet_all)
# print("max |diff| UNet:", max_diff_unet_all)

# %% cell 32
# # === Только mid_block ===

# cfg = VQConfig(bits=8, d=16, max_iter=25, on_cpu=False)

# unet_q_mid = quantize_unet_all_weights(pipe.unet.mid_block, cfg)

# pipe_fp = pipe
# pipe_q  = copy.deepcopy(pipe_fp)
# pipe_q.unet.mid_block = unet_q_mid

# device = torch.device("cuda")
# dtype  = torch.float16

# pipe_fp.unet.to(device, dtype=dtype)
# pipe_q.unet.to(device, dtype=dtype)

# pipe_fp.text_encoder.to(device, dtype=dtype)
# pipe_q.text_encoder.to(device, dtype=dtype)

# pipe_fp.unet.eval()
# pipe_q.unet.eval()
# pipe_fp.text_encoder.eval()
# pipe_q.text_encoder.eval()

# H = W = getattr(pipe_fp.unet.config, "sample_size", 64)
# B = 1

# g = torch.Generator(device=device).manual_seed(1234)
# sample = torch.randn(
#     B, pipe_fp.unet.config.in_channels, H, W,
#     generator=g, device=device, dtype=dtype
# )
# t = torch.randint(0, 1000, (B,), generator=g, device=device, dtype=torch.long)

# prompt = "a photo of a cute cat, high quality, 4k"
# text_inputs = pipe_fp.tokenizer(
#     [prompt],
#     padding="max_length",
#     max_length=pipe_fp.tokenizer.model_max_length,
#     return_tensors="pt"
# )
# input_ids = text_inputs.input_ids.to(device)

# with torch.no_grad():
#     encoder_hidden_states = pipe_fp.text_encoder(input_ids)[0].to(dtype)

#     out_fp = pipe_fp.unet(sample, t, encoder_hidden_states=encoder_hidden_states).sample
#     out_q  = pipe_q.unet(sample, t, encoder_hidden_states=encoder_hidden_states).sample

# diff = (out_fp.float() - out_q.float())
# rel = diff.norm().item() / (out_fp.float().norm().item() + 1e-12)
# mx  = diff.abs().max().item()

# print("UNet rel L2 error (forward probe):", rel)
# print("UNet max |diff|:", mx)

# %% cell 33
# num_inference_steps = 50
# guidance_scale = 5.0
# seed = 1234

# img_fp = generate_image(pipe_fp, prompt, seed,
#                         num_inference_steps=num_inference_steps,
#                         guidance_scale=guidance_scale)

# res = evaluate_against_fp(
#     img_fp_pil=img_fp,
#     pipe_test=pipe_q,
#     prompt=prompt,
#     clip_model=clip_model,
#     clip_tokenizer=clip_tokenizer,
#     lpips_model=lpips_model,
#     seed=seed,
#     num_inference_steps=num_inference_steps,
#     guidance_scale=guidance_scale,
# )

# img_q = res["img_test"]

# print(f"CLIP text–image (fp):    {res['clip_text_fp']:.4f}")
# print(f"CLIP text–image (quant): {res['clip_text_test']:.4f}")
# print(f"LPIPS (fp vs quant):     {res['lpips']:.4f}")
# print(f"CLIP image–image:        {res['clip_img_sim']:.4f}")

# visualize_comparison(
#     img_fp=img_fp,
#     img_q=img_q,
#     lpips_value=res["lpips"],
#     clip_img_sim=res["clip_img_sim"],
#     title="UNet mid_block quantization (Linear-only)"
# )

# %% [markdown] cell 34
# <h1>Эксперименты</h1>

# %% cell 35
# === Промпты для сравнения RVQ-вариантов ===
# PickScore train/val CSVs are the only prompt source for comparable runs.
# See load_pickscore_reference_prompts(...) below.


# %% cell 36
# === Shared experiment settings ===

# Match the VQDM reference runner: DDIM, guidance=5.0, 50 denoising steps.
num_inference_steps = 50
guidance_scale = 5.0
seed = 1234

# Fast smoke mode: one light layer, tiny codebook, short generation.
# Keep False for the real Hadamard/bpp sweep.
rvq_fast_smoke_test = False

HARDCODED_RVQ_LINEAR_TARGET_NAMES = [
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

HARDCODED_RVQ_CONV_TARGET_NAMES = [
    "mid_block.attentions.0.proj_in",
    "mid_block.attentions.0.proj_out",
    "mid_block.resnets.0.conv1",
    "mid_block.resnets.0.conv2",
    "mid_block.resnets.1.conv1",
    "mid_block.resnets.1.conv2",
]

# Current experiment target modes:
# Override with:
#   RVQ_TARGET_MODE=all_midblock RVQ_INCLUDE_CONV=1       # full mid_block, 18 Linear/Conv2d layers
#   RVQ_TARGET_MODE=hardcoded_midblock RVQ_INCLUDE_CONV=1 # VQDM-style 16 layers, skips time_emb_proj
#   RVQ_TARGET_MODE=all_midblock_linear                   # VQDM-style Linear-only, skips time_emb_proj
#   RVQ_TARGET_MODE=single_largest_linear
#   RVQ_TARGET_MODE=single_named_linear RVQ_SINGLE_TARGET_LAYER=<full layer name>
rvq_memory_safe_full_test = False
rvq_target_mode = RVQ_TARGET_MODE
rvq_single_target_layer_name = (
    RVQ_SINGLE_TARGET_LAYER_ENV
    or "mid_block.attentions.0.transformer_blocks.0.ff.net.0.proj"  # Linear (10240, 1280)
)
rvq_budgeted_funnel = bool(RVQ_FUNNEL_SWEEP)
if rvq_budgeted_funnel:
    rvq_target_mode = "hardcoded_midblock"
valid_rvq_target_modes = {"hardcoded_midblock", "all_midblock", "all_midblock_linear", "single_largest_linear", "single_named_linear"}
if rvq_target_mode not in valid_rvq_target_modes:
    raise ValueError(
        f"Unsupported RVQ_TARGET_MODE={rvq_target_mode!r}; "
        f"expected one of {sorted(valid_rvq_target_modes)}"
    )

H = W = getattr(pipe.unet.config, "sample_size", 64)
B = 1

# Linear-only GPU-first mode. The threshold is hardware-profile driven:
# H100/A100-like cards default to 40 GiB, V100-32GB-like cards default to 24 GiB.
# If memory is not enough, switch the flags below from top to bottom.
# 1) RVQ_GPU_TARGET_MEM_FRACTION=0.50, 2) rvq_keep_eval_pipes_on_device=False,
# 3) rvq_offload_after_xtx=True, 4) rvq_xtx_store_device="cpu", 5) rvq_final_device="cpu", 6) rvq_quant_on_cpu=True.
rvq_high_vram_gpu = _is_big_gpu(RVQ_HIGH_VRAM_THRESHOLD_GIB)
rvq_collect_device = runtime_device
rvq_xtx_store_device = runtime_device if rvq_high_vram_gpu else ("cpu" if globals().get("IN_KAGGLE", False) else runtime_device)
rvq_quant_on_cpu = False
rvq_final_device = runtime_device
rvq_final_dtype = torch.float32 if rvq_final_device == "cpu" else runtime_dtype
rvq_eval_device = runtime_device
rvq_keep_eval_pipes_on_device = True if rvq_high_vram_gpu else (False if globals().get("IN_KAGGLE", False) else True)
rvq_offload_after_xtx = False if rvq_high_vram_gpu else (True if globals().get("IN_KAGGLE", False) else False)
# Legacy CPU offload/deepcopy path is useful on weak/low-VRAM GPUs, but it is counterproductive on high-VRAM cards:
# it adds CPU<->GPU copies and triggers diffusers warnings for fp16 pipelines moved to CPU.
rvq_cpu_deepcopy_before_quant = _env_bool("RVQ_CPU_DEEPCOPY_BEFORE_QUANT", False if rvq_high_vram_gpu else True)
rvq_cpu_copy_before_pipe_clone = _env_bool("RVQ_CPU_COPY_BEFORE_PIPE_CLONE", False if rvq_high_vram_gpu else True)
# High-VRAM fast path: do not deepcopy the whole StableDiffusionPipeline for every sweep variant.
# Instead, temporarily replace only pipe.unet.mid_block with the quantized copy during evaluation,
# then restore the original FP mid_block. Disable with RVQ_EVAL_INPLACE_MIDBLOCK=0.
rvq_eval_inplace_midblock = _env_bool("RVQ_EVAL_INPLACE_MIDBLOCK", True if rvq_high_vram_gpu else False)
rvq_show_inner_progress = True
rvq_include_conv = bool(RVQ_INCLUDE_CONV_ENV)
if rvq_budgeted_funnel:
    rvq_include_conv = False
if RVQ_STORE_COMPRESSED_ENV is None:
    # On a high-VRAM GPU, dense materialized eval is much faster: compressed forward
    # reconstructs the dense weight from RVQ codes on every UNet call.
    rvq_store_compressed = False if rvq_high_vram_gpu else True
else:
    rvq_store_compressed = _env_bool("RVQ_STORE_COMPRESSED", True)

generation_device = rvq_eval_device
generation_dtype = runtime_dtype if str(generation_device).startswith("cuda") else torch.float32
generation_keep_on_device = rvq_keep_eval_pipes_on_device
rvq_generation_batch_size = RVQ_GEN_BATCH_ENV if RVQ_GEN_BATCH_ENV > 0 else ((32 if _AUTO_HUGE_GPU else 8) if rvq_high_vram_gpu else 1)
optimize_pipeline_for_inference(pipe, label="FP pipeline initial")

# VQDM/AQLM-like bpp profiles. Keep K=256 by default and change the number
# of RVQ stages M, matching the VQDM bpp schedule. RVQ_K_VALUES can add
# K ablations, e.g. RVQ_K_VALUES=128,256,512.
# The bpp below is paper-style packed-index bpp; PyTorch buffers are not bit-packed.
rvq_k_values_env = os.environ.get("RVQ_K_VALUES", "256")
rvq_vector_dims = [8]
rvq_reference_bpp_targets = [
    ("4.15", "target~4.15bpp", 4.15),
    ("3.15", "target~3.15bpp", 3.15),
    ("2.15", "target~2.15bpp", 2.15),
]


def _parse_rvq_k_values(value: str) -> List[int]:
    parts = [part.strip() for part in str(value).replace(";", ",").split(",")]
    values: List[int] = []
    for part in parts:
        if not part:
            continue
        K = int(part)
        if K < 2:
            raise ValueError(f"RVQ_K_VALUES entries must be >= 2, got {K}")
        if K not in values:
            values.append(K)
    if not values:
        raise ValueError("RVQ_K_VALUES must contain at least one integer K")
    return values


def _ceil_log2_int(value: int) -> int:
    value = int(value)
    if value <= 1:
        return 1
    return int(value - 1).bit_length()


def _nearest_rvq_stage_count(reference_bpp: float, *, d: int, bits: int) -> int:
    raw = float(reference_bpp) * int(d) / max(int(bits), 1)
    lo = max(1, math.floor(raw))
    hi = max(1, math.ceil(raw))
    return min((lo, hi), key=lambda m: (abs((m * int(bits) / int(d)) - float(reference_bpp)), m))


def _build_rvq_bpp_profiles(
    k_values: Sequence[int],
    *,
    vector_dims: Sequence[int] = (8,),
    reference_targets: Optional[Sequence[Tuple[str, str, float]]] = None,
) -> List[Dict[str, Any]]:
    reference_targets = list(reference_targets or rvq_reference_bpp_targets)
    k_values = [int(K) for K in k_values]
    profiles: List[Dict[str, Any]] = []
    for target_key, target_label, reference_bpp in reference_targets:
        for d_value in vector_dims:
            for K_value in k_values:
                bits_value = _ceil_log2_int(K_value)
                M_value = _nearest_rvq_stage_count(reference_bpp, d=int(d_value), bits=bits_value)
                stage_ks = tuple(int(K_value) for _ in range(M_value))
                stage_bits = _stage_bits_for_ks(stage_ks)
                key = f"{target_key}-d{int(d_value)}"
                label = f"{target_label} d={int(d_value)}"
                if len(k_values) > 1 or K_value != 256:
                    key += f"-k{K_value}"
                    label += f" K={K_value}"
                profiles.append({
                    "key": key,
                    "label": label,
                    "reference_bpp": float(reference_bpp),
                    "bits": bits_value,
                    "K": K_value,
                    "stage_Ks": stage_ks,
                    "stage_bits": stage_bits,
                    "d": int(d_value),
                    "rvq_M": M_value,
                })
    return profiles


RVQ_STAGE_K_PROFILES_BY_TARGET: Dict[str, List[Tuple[int, ...]]] = {
    "4.15": [
        (256, 256, 256, 256),
        (512, 256, 128, 128),
        (512, 512, 128, 128),
    ],
    "3.15": [
        (512, 256, 128),
        (256, 256, 256),
        (128, 256, 512),
        (1024, 128, 128),
        (512, 512, 64),
    ],
    "2.15": [
        (64, 1024),
        (512, 128),
        (128, 512),
        (1024, 64),
        (256, 256),
    ],
}


def _build_stagewise_rvq_bpp_profiles(
    *,
    vector_dims: Sequence[int] = (8,),
    reference_targets: Optional[Sequence[Tuple[str, str, float]]] = None,
) -> List[Dict[str, Any]]:
    reference_targets = list(reference_targets or rvq_reference_bpp_targets)
    profiles: List[Dict[str, Any]] = []
    for target_key, target_label, reference_bpp in reference_targets:
        stage_profiles = RVQ_STAGE_K_PROFILES_BY_TARGET.get(str(target_key), [])
        if not stage_profiles:
            raise ValueError(f"No stage-wise K profiles configured for target {target_key}")
        for d_value in vector_dims:
            for stage_ks_raw in stage_profiles:
                stage_ks = tuple(int(k) for k in stage_ks_raw)
                stage_bits = _stage_bits_for_ks(stage_ks)
                bits_value = max(stage_bits)
                key = f"{target_key}-d{int(d_value)}-ks" + "x".join(str(k) for k in stage_ks)
                label = f"{target_label} d={int(d_value)} Ks=" + "x".join(str(k) for k in stage_ks)
                profiles.append({
                    "key": key,
                    "label": label,
                    "reference_bpp": float(reference_bpp),
                    "bits": bits_value,
                    "K": max(stage_ks),
                    "stage_Ks": stage_ks,
                    "stage_bits": stage_bits,
                    "d": int(d_value),
                    "rvq_M": len(stage_ks),
                })
    return profiles


rvq_k_values = _parse_rvq_k_values(rvq_k_values_env)
if RVQ_STAGE_K_SWEEP:
    rvq_bpp_profiles = _build_stagewise_rvq_bpp_profiles(vector_dims=rvq_vector_dims)
else:
    rvq_bpp_profiles = _build_rvq_bpp_profiles(rvq_k_values, vector_dims=rvq_vector_dims)
seen_profile_keys = set()
for profile in rvq_bpp_profiles:
    key = str(profile["key"])
    if key in seen_profile_keys:
        raise ValueError(f"Duplicate RVQ bpp profile key: {key}")
    seen_profile_keys.add(key)
    if int(profile["bits"]) < _ceil_log2_int(int(profile["K"])):
        raise ValueError(f"RVQ profile {key} has too few bits for K={profile['K']}")


# Keep these aliases for older helper cells that expect scalar RVQ settings.
rvq_bits = rvq_bpp_profiles[0]["bits"]
rvq_d = rvq_bpp_profiles[0]["d"]
rvq_M = rvq_bpp_profiles[0]["rvq_M"]
rvq_reference_bpp = rvq_bpp_profiles[0]["reference_bpp"]

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
    csv_path = _resolve_pickscore_csv(filename, env_name="RVQ_PICKSCORE_DATA_DIR")

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


# Full Hadamard sweep controls. Keep the prompt protocol identical to VQDM.
rvq_use_reference_pickscore_prompts = True
rvq_calibration_prompt_count = 256
rvq_hadamard_sweep_prompt_count = 32
rvq_final_comparison_prompt_count = 32
rvq_hadamard_sweep_seeds = [1234, 4321, 999]
rvq_final_comparison_seeds = [1234, 4321, 999]
rvq_hadamard_block_candidates = [8, 16, 32, 64, 128, 256]
rvq_hadamard_sign_candidates = [False]  # previous sweep showed signs=False is consistently better here


def _resolve_rvq_hadamard_mode(mode: str, default_blocks: Sequence[int]) -> Tuple[List[Optional[int]], bool, str]:
    mode = str(mode or "auto").strip().lower()
    aliases = {"none": "noh", "no": "noh", "no_h": "noh", "no-h": "noh", "nohadamard": "noh"}
    mode = aliases.get(mode, mode)
    if mode in {"", "auto"}:
        return [int(b) for b in default_blocks], False, "auto"
    if mode == "noh":
        return [None], True, "noh"
    if mode.startswith("h") and mode[1:].isdigit():
        block = int(mode[1:])
        if block <= 0 or (block & (block - 1)) != 0:
            raise ValueError(f"RVQ_HADAMARD_MODE={mode!r} has invalid block size; expected h8/h16/h32/etc.")
        return [block], True, f"h{block}"
    raise ValueError("RVQ_HADAMARD_MODE must be one of: auto, noh, h8, h16, h32")


rvq_xtx_full_steps = num_inference_steps
rvq_xtx_seed = 0
rvq_kmeans_max_iter = 25
rvq_kmeans_n_init = 4
rvq_xtx_batch_size = _env_int("RVQ_XTX_BATCH", 0)       # 0 = auto
rvq_batch_assign = _env_int("RVQ_ASSIGN_BATCH", 0)      # 0 = auto
seeds = rvq_final_comparison_seeds
rvq_compute_hps = bool(RVQ_COMPUTE_HPS_ENV)

log_section("RVQ experiment settings")
print("RVQ bpp profiles:")
for p in rvq_bpp_profiles:
    p_stage_ks = tuple(int(k) for k in p.get("stage_Ks", [p["K"]] * int(p["rvq_M"])))
    p_stage_bits = tuple(int(b) for b in p.get("stage_bits", [_ceil_log2_int(int(p["K"]))] * int(p["rvq_M"])))
    print(
        f"  {p['label']}: M={len(p_stage_ks)}, Ks={_format_stage_ks(p_stage_ks)}, "
        f"bits={list(p_stage_bits)}, d={p['d']}, "
        f"index_bpp={sum(p_stage_bits) / p['d']:.2f}, VQDM_ref~{p['reference_bpp']}"
    )
print(f"stage-wise K sweep: {RVQ_STAGE_K_SWEEP}")
print(f"Hadamard mode env: {RVQ_HADAMARD_MODE_ENV}")
print(f"target mode: {rvq_target_mode}")
if rvq_budgeted_funnel:
    print(
        "budgeted funnel: ON | "
        f"K_values={RVQ_FUNNEL_K_VALUES_ENV}, block_topk={RVQ_FUNNEL_BLOCK_TOPK}, "
        f"final_topk_per_bpp_per_H_noH={RVQ_FUNNEL_FINAL_TOPK}, include_conv_stage={RVQ_FUNNEL_INCLUDE_CONV_STAGE}, "
        f"conv_topk={RVQ_FUNNEL_CONV_TOPK}, results_json={RVQ_FUNNEL_RESULTS_JSON}"
    )
if rvq_target_mode == "single_named_linear":
    print(f"single target layer: {rvq_single_target_layer_name}")
print(f"include conv targets: {rvq_include_conv}")
print(
    f"hw_profile={RVQ_HW_PROFILE}, cuda_totals_gib={[round(x, 1) for x in _AUTO_CUDA_TOTALS_GIB]}, "
    f"high_vram_gpu={rvq_high_vram_gpu}, high_vram_threshold={RVQ_HIGH_VRAM_THRESHOLD_GIB:.1f}, "
    f"gpu_total_gib={_cuda_total_gib():.1f}"
)
print(f"collect={rvq_collect_device}, xtx={rvq_xtx_store_device}, quant_on_cpu={rvq_quant_on_cpu}, final={rvq_final_device}")
print(f"cpu_deepcopy_before_quant={rvq_cpu_deepcopy_before_quant}, cpu_copy_before_pipe_clone={rvq_cpu_copy_before_pipe_clone}")
print(f"eval_inplace_midblock={rvq_eval_inplace_midblock}")
print(
    f"multi_gpu_assign={RVQ_USE_MULTI_GPU_ASSIGN}, "
    f"multi_gpu_xtx={RVQ_USE_MULTI_GPU_XTX}, "
    f"multi_gpu_min_N={RVQ_MULTI_GPU_MIN_N}, devices={RVQ_MULTI_GPU_DEVICES_ENV or 'auto'}, "
    f"xtx_devices={RVQ_XTX_DEVICES_ENV or 'auto'}"
)
print(f"Hadamard default candidates: blocks={rvq_hadamard_block_candidates}, signs={rvq_hadamard_sign_candidates}")
print(f"XTX calibration seed: {rvq_xtx_seed}")
print(
    f"k-means: max_iter={rvq_kmeans_max_iter}, n_init={rvq_kmeans_n_init} per RVQ stage, "
    f"batch_assign={'auto' if rvq_batch_assign <= 0 else rvq_batch_assign}, "
    f"xtx_batch={'auto' if rvq_xtx_batch_size <= 0 else rvq_xtx_batch_size}, "
    f"gpu_target_mem_fraction={RVQ_GPU_TARGET_MEM_FRACTION}"
)
print(
    f"eval acceleration: gen_batch={rvq_generation_batch_size}, compute_hps={rvq_compute_hps}, "
    f"store_compressed={rvq_store_compressed}, materialize_compressed_for_eval={RVQ_MATERIALIZE_COMPRESSED_FOR_EVAL}"
)
print(
    f"full-XTX recode: always on when XTX is available, passes={RVQ_FULL_XTX_RECODE_PASSES}, "
    f"row_batch={'all' if RVQ_FULL_XTX_RECODE_ROW_BATCH <= 0 else RVQ_FULL_XTX_RECODE_ROW_BATCH}, "
    f"max_cols={'uncapped' if RVQ_FULL_XTX_RECODE_MAX_COLS <= 0 else RVQ_FULL_XTX_RECODE_MAX_COLS}, "
    f"log_xtx_block_energy={RVQ_LOG_XTX_BLOCK_ENERGY}"
)
log_cuda_memory("after RVQ settings")

if rvq_fast_smoke_test:
    num_inference_steps = 6
    print("RVQ fast smoke mode is ON: quick functionality check, not comparable quality numbers.")
else:
    if rvq_target_mode == "hardcoded_midblock":
        print("RVQ hardcoded-mid_block bpp sweep is ON: fixed 10 Linear layers; set RVQ_INCLUDE_CONV=1 to add 6 VQDM-style Conv2d layers.")
    elif rvq_target_mode == "all_midblock":
        print("RVQ all-mid_block bpp sweep is ON: all Linear layers plus Conv2d when RVQ_INCLUDE_CONV=1; this is the 18-layer full-midblock mode.")
    elif rvq_target_mode == "all_midblock_linear":
        print("RVQ all-mid_block-Linear bpp sweep is ON: all VQDM-style Linear layers, no-sign Hadamard blocks, d=8 profiles.")
    elif rvq_target_mode == "single_largest_linear":
        print("RVQ single-largest-layer bpp sweep is ON: one Linear layer, no-sign Hadamard blocks, d=8 profiles.")
    else:
        print("RVQ single-named-layer bpp sweep is ON: one requested Linear layer, no-sign Hadamard blocks, d=8 profiles.")

# %% cell 37
# === Hadamard/bpp sweep setup: target, XTX, FP cache ===

log_section("RVQ setup: target layers, XTX, FP image cache")
rvq_vqdm_target_layers_all = print_mid_block_quant_layers(
    pipe.unet,
    min_features=16,
    include_linear=True,
    include_conv=True,
)
rvq_all_midblock_layer_registry = print_all_mid_block_quant_layers(
    pipe.unet,
    min_features=16,
    include_linear=True,
    include_conv=True,
)

rvq_target_layers_all = rvq_all_midblock_layer_registry
rvq_all_midblock_target_layers_all = {
    name: layer for name, layer in rvq_all_midblock_layer_registry.items()
    if rvq_include_conv or not isinstance(layer, nn.Conv2d)
}
vqdm_linear_target_layers_all = {
    name: layer for name, layer in rvq_vqdm_target_layers_all.items()
    if isinstance(layer, nn.Linear)
}
linear_target_layers_all = {
    name: layer for name, layer in rvq_all_midblock_layer_registry.items()
    if isinstance(layer, nn.Linear)
}
if not linear_target_layers_all:
    raise RuntimeError("No Linear target layers found in mid_block")

if rvq_target_mode == "hardcoded_midblock":
    hardcoded_names = list(HARDCODED_RVQ_LINEAR_TARGET_NAMES)
    if rvq_include_conv:
        hardcoded_names += list(HARDCODED_RVQ_CONV_TARGET_NAMES)
    missing = [name for name in hardcoded_names if name not in rvq_target_layers_all]
    if missing:
        raise RuntimeError(
            "Hardcoded RVQ target layer(s) were not found in mid_block: "
            + ", ".join(missing)
        )
    rvq_target_layers = {name: rvq_target_layers_all[name] for name in hardcoded_names}
elif rvq_target_mode == "all_midblock":
    rvq_target_layers = dict(rvq_all_midblock_target_layers_all)
elif rvq_target_mode == "all_midblock_linear":
    rvq_target_layers = dict(vqdm_linear_target_layers_all)
elif rvq_target_mode == "single_largest_linear":
    largest_name = max(linear_target_layers_all, key=lambda name: linear_target_layers_all[name].weight.numel())
    rvq_target_layers = {largest_name: linear_target_layers_all[largest_name]}
    print(f"Selected largest Linear target layer: {largest_name}")
elif rvq_target_mode == "single_named_linear":
    if rvq_single_target_layer_name in linear_target_layers_all:
        rvq_target_layers = {rvq_single_target_layer_name: linear_target_layers_all[rvq_single_target_layer_name]}
    else:
        largest_name = max(linear_target_layers_all, key=lambda name: linear_target_layers_all[name].weight.numel())
        rvq_target_layers = {largest_name: linear_target_layers_all[largest_name]}
        print(f"Requested target layer not found; selected largest Linear instead: {largest_name}")
else:
    raise AssertionError(f"Unhandled rvq_target_mode={rvq_target_mode!r}")

rvq_target_layer_names = list(rvq_target_layers.keys())
rvq_target_layer_set = set(rvq_target_layer_names)

print()
print("Selected RVQ target layers:")
for name, layer in rvq_target_layers.items():
    print(f"  {name:60s} {type(layer).__name__} weight={tuple(layer.weight.shape)} params={layer.weight.numel():,}")
print("Target layer count:", len(rvq_target_layer_names))
print("time_emb_proj included:", any("time_emb_proj" in name for name in rvq_target_layer_names))

if rvq_fast_smoke_test:
    rvq_calibration_prompts = load_pickscore_reference_prompts(1, seed=0, eval_mode=False)
    rvq_xtx_num_inference_steps = 6
    hadamard_sweep_prompts = load_pickscore_reference_prompts(1, seed=0, eval_mode=True)
    rvq_final_comparison_prompts = list(hadamard_sweep_prompts)
    hadamard_sweep_seeds = [1234]
    hadamard_block_candidates, rvq_fixed_transform_sweep, rvq_hadamard_mode = _resolve_rvq_hadamard_mode(
        RVQ_HADAMARD_MODE_ENV,
        [8],
    )
    hadamard_sign_candidates = [False]
    active_bpp_profiles = [{"key": "smoke", "label": "smoke~0.5bpp", "reference_bpp": float("nan"), "bits": 4, "K": 16, "d": 8, "rvq_M": 1}]
else:
    if not rvq_use_reference_pickscore_prompts:
        raise RuntimeError("RVQ prompt selection now requires PickScore CSVs; keep rvq_use_reference_pickscore_prompts=True.")
    rvq_calibration_prompts = load_pickscore_reference_prompts(
        rvq_calibration_prompt_count,
        seed=0,
        eval_mode=False,
    )
    hadamard_sweep_prompts = load_pickscore_reference_prompts(
        rvq_hadamard_sweep_prompt_count,
        seed=0,
        eval_mode=True,
    )
    rvq_final_comparison_prompts = list(hadamard_sweep_prompts[:rvq_final_comparison_prompt_count])
    print("Using required PickScore train/val prompts like the VQDM reference runner.")

    rvq_xtx_num_inference_steps = rvq_xtx_full_steps
    hadamard_sweep_seeds = list(rvq_hadamard_sweep_seeds)
    hadamard_block_candidates, rvq_fixed_transform_sweep, rvq_hadamard_mode = _resolve_rvq_hadamard_mode(
        RVQ_HADAMARD_MODE_ENV,
        rvq_hadamard_block_candidates,
    )
    hadamard_sign_candidates = list(rvq_hadamard_sign_candidates)
    active_bpp_profiles = [dict(p) for p in rvq_bpp_profiles]

print(
    "Resolved RVQ transform sweep: "
    f"mode={rvq_hadamard_mode}, fixed={rvq_fixed_transform_sweep}, "
    f"candidates={[('noH' if b is None else f'H{int(b)}') for b in hadamard_block_candidates]}"
)


def rvq_profile_key(profile: Dict[str, Any]) -> str:
    return str(profile["key"])


def rvq_profile_label(profile: Dict[str, Any]) -> str:
    return str(profile.get("label", rvq_profile_key(profile)))


def rvq_profile_K(profile: Dict[str, Any]) -> int:
    return int(profile.get("K", 1 << int(profile["bits"])))


def rvq_profile_stage_ks(profile: Dict[str, Any]) -> Tuple[int, ...]:
    if "stage_Ks" in profile:
        return tuple(int(k) for k in profile["stage_Ks"])
    return tuple(rvq_profile_K(profile) for _ in range(max(int(profile["rvq_M"]), 1)))


def rvq_profile_stage_bits(profile: Dict[str, Any]) -> Tuple[int, ...]:
    if "stage_bits" in profile:
        return tuple(int(b) for b in profile["stage_bits"])
    return _stage_bits_for_ks(rvq_profile_stage_ks(profile))


def format_transform_label(block: Optional[int], with_signs: Optional[bool] = None) -> str:
    if block is None:
        return "noH"
    if with_signs is None:
        return f"H{int(block)}"
    return f"H{int(block)} signs={bool(with_signs)}"


def make_sweep_spec(profile: Dict[str, Any], block: Optional[int], with_signs: Optional[bool]) -> Dict[str, Any]:
    stage_ks = rvq_profile_stage_ks(profile)
    stage_bits = rvq_profile_stage_bits(profile)
    return {
        "profile": dict(profile),
        "profile_key": rvq_profile_key(profile),
        "profile_label": rvq_profile_label(profile),
        "reference_bpp": float(profile.get("reference_bpp", float("nan"))),
        "bits": int(max(stage_bits)),
        "K": max(stage_ks),
        "stage_Ks": list(stage_ks),
        "stage_bits": list(stage_bits),
        "d": int(profile["d"]),
        "rvq_M": len(stage_ks),
        "index_bpp": sum(stage_bits) / int(profile["d"]),
        "block": None if block is None else int(block),
        "with_signs": None if block is None or with_signs is None else bool(with_signs),
        "transform_family": "noH" if block is None else "H",
        "transform_label": format_transform_label(None if block is None else int(block), with_signs),
    }


def profile_config_key(item: Dict[str, Any]) -> tuple:
    profile_key = str(item.get("profile_key", item.get("key")))
    stage_ks = tuple(int(k) for k in item.get("stage_Ks", []))
    if not stage_ks:
        stage_ks = tuple(rvq_profile_K(item) for _ in range(max(int(item["rvq_M"]), 1)))
    return (profile_key, tuple(stage_ks), int(item["d"]))


def sweep_spec_key(spec: Dict[str, Any]) -> tuple:
    # Include bits/K/d/M so stale hardcoded results from an older bpp scheme cannot be reused silently.
    block_key = None if spec.get("block") is None else int(spec["block"])
    signs_key = None if spec.get("with_signs") is None else bool(spec["with_signs"])
    return profile_config_key(spec) + (block_key, signs_key)


def completed_sweep_keys(results: Sequence[Dict[str, Any]]) -> set[tuple]:
    keys = set()
    for result in results:
        try:
            keys.add(sweep_spec_key(result))
        except (KeyError, TypeError, ValueError):
            pass
    return keys


def result_matches_profile(result: Dict[str, Any], profile: Dict[str, Any]) -> bool:
    try:
        return profile_config_key(result) == (
            rvq_profile_key(profile),
            tuple(rvq_profile_stage_ks(profile)),
            int(profile["d"]),
        )
    except (KeyError, TypeError, ValueError):
        return False


sweep_specs = [
    make_sweep_spec(profile, block, with_signs)
    for block in hadamard_block_candidates
    for profile in active_bpp_profiles
    for with_signs in hadamard_sign_candidates
]
split_at = (len(sweep_specs) + 1) // 2
hadamard_sweep_specs_part1 = sweep_specs[:split_at]
hadamard_sweep_specs_part2 = sweep_specs[split_at:]
print(
    f"Hadamard/bpp sweep setup: {len(sweep_specs)} variants "
    f"({len(hadamard_sweep_specs_part1)} + {len(hadamard_sweep_specs_part2)}) | "
    f"profiles={len(active_bpp_profiles)} | mode={rvq_hadamard_mode} | "
    f"transforms={[format_transform_label(b, None if b is None else False) for b in hadamard_block_candidates]} | "
    f"signs={hadamard_sign_candidates} | fixed_transform={rvq_fixed_transform_sweep} | "
    f"calibration_prompts={len(rvq_calibration_prompts)} | "
    f"sweep_eval_pairs={len(hadamard_sweep_prompts) * len(hadamard_sweep_seeds)} | "
    f"xtx_steps={rvq_xtx_num_inference_steps} | eval_steps={num_inference_steps} | "
    f"kmeans_n_init={rvq_kmeans_n_init}"
)


def make_rvq_aw_cfg(
    use_hadamard: bool,
    block: int | None = None,
    with_signs: bool = False,
    profile: Optional[Dict[str, Any]] = None,
) -> VQConfig:
    profile = dict(profile or active_bpp_profiles[0])
    profile_stage_ks = rvq_profile_stage_ks(profile)
    profile_stage_bits = rvq_profile_stage_bits(profile)
    cfg_bits = 4 if rvq_fast_smoke_test else int(max(profile_stage_bits))
    cfg_K = (1 << cfg_bits) if rvq_fast_smoke_test else int(max(profile_stage_ks))
    cfg_d = int(profile["d"])
    cfg_stage_ks = (cfg_K,) if rvq_fast_smoke_test else tuple(profile_stage_ks)
    cfg_M = 1 if rvq_fast_smoke_test else len(cfg_stage_ks)
    cfg_max_iter = 4 if rvq_fast_smoke_test else rvq_kmeans_max_iter
    cfg_n_init = 1 if rvq_fast_smoke_test else rvq_kmeans_n_init
    cfg = VQConfig(bits=cfg_bits, d=cfg_d, max_iter=cfg_max_iter, on_cpu=rvq_quant_on_cpu)
    cfg.K_override = cfg_K
    cfg.stage_Ks = tuple(cfg_stage_ks)
    cfg.use_activation_aware = True
    cfg.use_hadamard = use_hadamard
    cfg.group_channels = True
    cfg.rvq_M = cfg_M
    cfg.rvq_first_stage_n_init = cfg_n_init
    cfg.batch_assign = rvq_batch_assign
    cfg.free_xtx_after_layer = False  # keep XTX for all sweep and final-comparison variants
    cfg.show_inner_progress = rvq_show_inner_progress
    cfg.store_compressed = rvq_store_compressed
    cfg.log_xtx_block_energy = RVQ_LOG_XTX_BLOCK_ENERGY
    cfg.full_xtx_recode_passes = RVQ_FULL_XTX_RECODE_PASSES
    cfg.full_xtx_recode_row_batch = RVQ_FULL_XTX_RECODE_ROW_BATCH
    cfg.full_xtx_recode_max_cols = RVQ_FULL_XTX_RECODE_MAX_COLS
    if block is not None:
        cfg.hadamard_block = int(block)
    cfg.hadamard_with_signs = bool(with_signs)
    return cfg


def hadamard_rank_score(agg: Dict[str, float]) -> float:
    return float(agg["clip_ii"]) - float(agg["lpips"])


@torch.no_grad()
def eval_with_temporary_quantized_midblock(
    *,
    pipe_obj,
    unet_q: nn.Module,
    prompts,
    seeds,
    fp_cache,
    image_pbar=None,
    eval_label: str = "RVQ",
):
    """Evaluate a quantized mid_block without deepcopying the whole pipeline."""
    missing = count_missing_fp_images(
        fp_cache,
        prompts,
        seeds,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
    )
    if missing:
        log_event(f"FP cache missing {missing} image(s) before in-place eval; precomputing before patch")
        precompute_fp_image_cache(
            pipe_obj,
            prompts,
            seeds,
            cache=fp_cache,
            image_pbar=image_pbar,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
        )

    fp_mid_block = pipe_obj.unet.mid_block
    fp_was_training = pipe_obj.unet.training
    forward_drift_metrics = compute_forward_drift_with_midblock_patch(
        pipe_obj,
        unet_q,
        prompts,
        eval_label=eval_label,
    )
    try:
        pipe_obj.unet.mid_block = unet_q.to(rvq_eval_device, dtype=rvq_final_dtype)
        optimize_pipeline_for_inference(pipe_obj, label=f"{eval_label} in-place patched pipeline")
        log_event(f"Evaluating {eval_label} by in-place mid_block patch; full pipeline deepcopy skipped")
        log_cuda_memory(f"after in-place mid_block patch for {eval_label}")
        return eval_prompts_against_fp(
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
        log_cuda_memory(f"after restoring FP mid_block for {eval_label}")


def build_rvq_midblock_pipe_from_cfg(
    cfg: VQConfig,
    label: str,
    *,
    target_layer_names: Optional[set[str]] = None,
    xtx_stats_override: Optional[Dict[str, Any]] = None,
):
    target_layer_names = set(target_layer_names or rvq_target_layer_set)
    xtx_stats_local = rvq_xtx_stats if xtx_stats_override is None else xtx_stats_override
    log_section(f"Quantizing {label}")
    log_event(
        f"RVQ config: M={len(_cfg_stage_ks(cfg))}, stage_Ks={_format_stage_ks(_cfg_stage_ks(cfg))}, "
        f"stage_bits={list(_cfg_stage_bits(cfg))}, d={cfg.d}, "
        f"hadamard={cfg.use_hadamard}, block={cfg.hadamard_block}, signs={cfg.hadamard_with_signs}, "
        f"activation_aware={cfg.use_activation_aware}, full_xtx_recode_passes={cfg.full_xtx_recode_passes}, "
        f"quant_on_cpu={cfg.on_cpu}"
    )
    log_cuda_memory("before quantization")
    t0 = time.time()
    unet_q = quantize_unet_all_weights(
        pipe.unet.mid_block,
        cfg=cfg,
        xtx_stats=xtx_stats_local,
        target_layer_names=target_layer_names,
        prefix="mid_block",
        final_device=rvq_final_device,
        final_dtype=rvq_final_dtype,
        cpu_deepcopy_before_quant=rvq_cpu_deepcopy_before_quant,
    )
    print(f"{label} quantized in {time.time() - t0:.1f}s")
    log_cuda_memory("after quantization")

    if RVQ_MATERIALIZE_COMPRESSED_FOR_EVAL:
        t_mat0 = time.time()
        n_mat = materialize_compressed_rvq_linears_for_inference(
            unet_q,
            device=rvq_final_device,
            dtype=rvq_final_dtype,
        )
        if n_mat:
            log_event(
                f"Materialized {n_mat} compressed RVQ Linear layer(s) for fast eval in {time.time() - t_mat0:.2f}s"
            )
            log_cuda_memory("after RVQ materialization")

    if rvq_eval_inplace_midblock:
        log_event("Skipping full pipeline clone; quantized mid_block will be evaluated in-place")
        log_cuda_memory("after quantized mid_block build")
        return None, unet_q

    pipe_base_device = rvq_eval_device if rvq_keep_eval_pipes_on_device else "cpu"
    pipe_base_dtype = rvq_final_dtype if pipe_base_device == rvq_eval_device else torch.float32

    if str(pipe_base_device).startswith("cuda"):
        if rvq_cpu_copy_before_pipe_clone:
            move_to_cpu_safely(pipe)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            pipe_q = force_pipeline_device_dtype(copy.deepcopy(pipe), pipe_base_device, pipe_base_dtype)
            if rvq_keep_eval_pipes_on_device:
                force_pipeline_device_dtype(pipe, pipe_base_device, pipe_base_dtype)
        else:
            if _module_device(pipe.unet).type != "cuda":
                force_pipeline_device_dtype(pipe, pipe_base_device, pipe_base_dtype)
            log_event("Cloning FP pipeline directly on CUDA; CPU pre-copy disabled")
            pipe_q = force_pipeline_device_dtype(copy.deepcopy(pipe), pipe_base_device, pipe_base_dtype)
    else:
        pipe_fp_local = move_to_cpu_safely(pipe)
        pipe_q = copy.deepcopy(pipe_fp_local)

    pipe_q.unet.mid_block = unet_q.to(pipe_base_device, dtype=pipe_base_dtype)
    optimize_pipeline_for_inference(pipe_q, label=label)
    if rvq_keep_eval_pipes_on_device:
        optimize_pipeline_for_inference(pipe, label="FP pipeline")

    if not rvq_keep_eval_pipes_on_device:
        move_to_cpu_safely(pipe)
        move_to_cpu_safely(pipe_q)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    log_cuda_memory("after quantized pipeline build")
    return pipe_q, unet_q


def run_hadamard_sweep_chunk(chunk_specs, chunk_label: str):
    if "rvq_xtx_stats" not in globals():
        raise RuntimeError("Run the Hadamard/bpp sweep setup cell first to collect rvq_xtx_stats")
    if "rvq_fp_image_cache" not in globals():
        raise RuntimeError("Run the Hadamard/bpp sweep setup cell first to precompute rvq_fp_image_cache")

    global hadamard_sweep_results, hadamard_sweep_completed_keys
    if "hadamard_sweep_results" not in globals():
        hadamard_sweep_results = []
    if "hadamard_sweep_completed_keys" not in globals():
        hadamard_sweep_completed_keys = completed_sweep_keys(hadamard_sweep_results)

    pending_specs = [spec for spec in chunk_specs if sweep_spec_key(spec) not in hadamard_sweep_completed_keys]

    if not pending_specs:
        print(f"{chunk_label}: all variants are already done.")
        return hadamard_sweep_results

    sweep_eval_pairs = len(hadamard_sweep_prompts) * len(hadamard_sweep_seeds)
    sweep_image_total = len(pending_specs) * sweep_eval_pairs  # only quantized images; FP images are cached already
    log_section(chunk_label)
    log_event(
        f"Pending variants={len(pending_specs)}, eval_pairs_per_variant={sweep_eval_pairs}, "
        f"total_generated_quantized_images={sweep_image_total}"
    )
    log_cuda_memory(f"before {chunk_label}")

    with tqdm(total=sweep_image_total, desc=f"{chunk_label} images", unit="image") as image_pbar:
        for spec in pending_specs:
            profile = spec["profile"]
            block = None if spec.get("block") is None else int(spec["block"])
            with_signs = None if spec.get("with_signs") is None else bool(spec["with_signs"])
            use_hadamard = block is not None
            cfg = make_rvq_aw_cfg(
                use_hadamard=use_hadamard,
                block=block,
                with_signs=False if with_signs is None else with_signs,
                profile=profile,
            )
            transform_label = format_transform_label(block, with_signs)
            label = f"RVQ-AW {spec['profile_label']} + {transform_label}"

            t_variant0 = time.perf_counter()
            t_quant0 = time.perf_counter()
            pipe_q, unet_q = build_rvq_midblock_pipe_from_cfg(cfg, label)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t_quant = time.perf_counter() - t_quant0

            t_eval0 = time.perf_counter()
            if pipe_q is None:
                agg, rows = eval_with_temporary_quantized_midblock(
                    pipe_obj=pipe,
                    unet_q=unet_q,
                    prompts=hadamard_sweep_prompts,
                    seeds=hadamard_sweep_seeds,
                    fp_cache=rvq_fp_image_cache,
                    image_pbar=image_pbar,
                    eval_label=label,
                )
            else:
                agg, rows = eval_prompts_against_fp(
                    pipe,
                    pipe_q,
                    hadamard_sweep_prompts,
                    hadamard_sweep_seeds,
                    fp_cache=rvq_fp_image_cache,
                    image_pbar=image_pbar,
                    eval_label=label,
                )
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t_eval = time.perf_counter() - t_eval0
            t_variant = time.perf_counter() - t_variant0
            log_event(f"Variant timing | {label} | quant_build={t_quant:.1f}s, eval={t_eval:.1f}s, total={t_variant:.1f}s")
            score = hadamard_rank_score(agg)

            result = {
                "label": label,
                "profile_key": spec["profile_key"],
                "profile_label": spec["profile_label"],
                "reference_bpp": spec["reference_bpp"],
                "index_bpp": spec["index_bpp"],
                "bits": spec["bits"],
                "K": spec["K"],
                "stage_Ks": spec.get("stage_Ks", []),
                "stage_bits": spec.get("stage_bits", []),
                "d": spec["d"],
                "rvq_M": spec["rvq_M"],
                "block": block,
                "with_signs": with_signs,
                "transform_family": spec.get("transform_family", "H" if use_hadamard else "noH"),
                "transform_label": transform_label,
                "score": score,
                "agg": agg,
                "rows": rows,
            }
            hadamard_sweep_results.append(result)
            hadamard_sweep_completed_keys.add(sweep_spec_key(spec))

            print(
                f"SWEEP RESULT | {label} | index_bpp={spec['index_bpp']:.2f} | "
                f"score={score:.6f} | agg={agg}"
            )

            del pipe_q, unet_q
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            log_cuda_memory(f"after {label}")

    log_cuda_memory(f"after {chunk_label}")
    return hadamard_sweep_results


print("Compression stats for selected target and bpp profiles:")
for profile in active_bpp_profiles:
    cfg_for_stats = make_rvq_aw_cfg(use_hadamard=False, profile=profile)
    stage_ks = rvq_profile_stage_ks(profile)
    stage_bits = rvq_profile_stage_bits(profile)
    print()
    print(
        f"--- {rvq_profile_label(profile)} | M={len(stage_ks)}, Ks={_format_stage_ks(stage_ks)}, "
        f"stage_bits={list(stage_bits)}, index_bpp={sum(stage_bits) / int(profile['d']):.3f}, d={profile['d']} ---"
    )
    compute_rvq_compression_for_layers(
        layers=rvq_target_layers,
        cfg=cfg_for_stats,
        fp_bits=16,
        codebook_bits=16,
        verbose=True,
    )

rvq_xtx_stats_signature = {
    "target_layer_names": tuple(rvq_target_layer_names),
    "calibration_prompts": tuple(rvq_calibration_prompts),
    "num_inference_steps": int(rvq_xtx_num_inference_steps),
    "guidance_scale": float(guidance_scale),
    "xtx_seed": int(rvq_xtx_seed),
    "include_conv": bool(rvq_include_conv),
}

reuse_existing_xtx = (
    "rvq_xtx_stats" in globals()
    and isinstance(rvq_xtx_stats, dict)
    and set(rvq_target_layer_names).issubset(set(rvq_xtx_stats.keys()))
    and globals().get("rvq_xtx_stats_signature_saved") == rvq_xtx_stats_signature
)

if reuse_existing_xtx:
    print()
    print("Reusing existing rvq_xtx_stats from this session.")
    log_cuda_memory("reusing XTX")
else:
    log_section("Collect XTX")
    print("Collecting XTX once for all Hadamard/bpp candidates...")
    log_event(
        f"calibration_prompts={len(rvq_calibration_prompts)}, steps={rvq_xtx_num_inference_steps}, "
        f"collect_device={rvq_collect_device}, xtx_store_device={rvq_xtx_store_device}, seed={rvq_xtx_seed}"
    )
    log_cuda_memory("before XTX collection")
    t0 = time.time()
    rvq_xtx_stats = collect_midblock_xtx_streaming_autobatch(
        pipe=pipe,
        calibration_prompts=rvq_calibration_prompts,
        layer_names=rvq_target_layer_names,
        xtx_batch_size=rvq_xtx_batch_size,
        num_inference_steps=rvq_xtx_num_inference_steps,
        guidance_scale=guidance_scale,
        collect_device=rvq_collect_device,
        xtx_store_device=rvq_xtx_store_device,
        accum_dtype=torch.float32,
        group_channels=True,
        show_progress=True,
        generator_seed=rvq_xtx_seed,
    )
    if rvq_offload_after_xtx:
        move_to_cpu_safely(pipe)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    rvq_xtx_stats_signature_saved = rvq_xtx_stats_signature
    print(f"XTX collected in {time.time() - t0:.1f}s")
    log_cuda_memory("after XTX collection")

rvq_fp_image_cache = globals().get("rvq_fp_image_cache", {})
rvq_final_cache_prompts = rvq_final_comparison_prompts if not rvq_fast_smoke_test else hadamard_sweep_prompts
rvq_final_cache_seeds = list(rvq_final_comparison_seeds) if not rvq_fast_smoke_test else hadamard_sweep_seeds
rvq_cache_prompt_map = {p: None for p in list(hadamard_sweep_prompts) + list(rvq_final_cache_prompts)}
rvq_cache_seed_map = {int(s): None for s in list(hadamard_sweep_seeds) + list(rvq_final_cache_seeds)}
rvq_cache_prompts = list(rvq_cache_prompt_map.keys())
rvq_cache_seeds = list(rvq_cache_seed_map.keys())

missing_fp = count_missing_fp_images(
    rvq_fp_image_cache,
    rvq_cache_prompts,
    rvq_cache_seeds,
    num_inference_steps=num_inference_steps,
    guidance_scale=guidance_scale,
)
log_section("Prepare FP image cache")
print(f"Precomputing/reusing FP images: missing={missing_fp}, cache_total_before={len(rvq_fp_image_cache)}")
log_event(
    f"cache_prompts={len(rvq_cache_prompts)}, cache_seeds={len(rvq_cache_seeds)}, "
    f"steps={num_inference_steps}, guidance_scale={guidance_scale}"
)
log_cuda_memory("before FP image cache")

if missing_fp:
    with tqdm(total=missing_fp, desc="RVQ FP image cache", unit="image") as image_pbar:
        precompute_fp_image_cache(
            pipe,
            rvq_cache_prompts,
            rvq_cache_seeds,
            cache=rvq_fp_image_cache,
            image_pbar=image_pbar,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
        )

print(f"FP image cache ready: {len(rvq_fp_image_cache)} images")
log_cuda_memory("after FP image cache")


def _profile_target_key(profile: Dict[str, Any]) -> str:
    return str(profile.get("key", "")).split("-", 1)[0]


def _funnel_transform_family(item: Dict[str, Any]) -> str:
    return "H" if item.get("block") is not None else "noH"


def _funnel_variant_key(result: Dict[str, Any]) -> tuple:
    stage_ks = tuple(int(k) for k in result.get("stage_Ks", []))
    if not stage_ks:
        stage_ks = tuple(int(result.get("K", 1 << int(result["bits"]))) for _ in range(int(result["rvq_M"])))
    return (
        str(result["profile_key"]),
        stage_ks,
        int(result["d"]),
        None if result.get("block") is None else int(result["block"]),
        None if result.get("with_signs") is None else bool(result["with_signs"]),
    )


def _dedupe_ranked_results(results: Sequence[Dict[str, Any]], topk: int) -> List[Dict[str, Any]]:
    best: Dict[tuple, Dict[str, Any]] = {}
    for result in sorted(results, key=lambda r: float(r["score"]), reverse=True):
        key = _funnel_variant_key(result)
        if key not in best:
            best[key] = result
    ranked = sorted(best.values(), key=lambda r: float(r["score"]), reverse=True)
    return ranked[:max(1, int(topk))]


def select_funnel_final_candidates(results: Sequence[Dict[str, Any]], topk: int) -> List[Dict[str, Any]]:
    """Select top-k inside each bpp and separately for H/noH tracks."""
    topk = max(1, int(topk))
    ranked = _dedupe_ranked_results(results, max(len(results), topk))

    selected: List[Dict[str, Any]] = []
    selected_keys: set[tuple] = set()
    # Print/evaluate low-bpp groups first; 4 bpp is still kept as a sanity reference.
    for target_key in ("2.15", "3.15", "4.15"):
        for family in ("H", "noH"):
            rows = [
                r for r in ranked
                if _profile_target_key(r["profile"]) == target_key
                and _funnel_transform_family(r) == family
            ]
            for result in rows[:topk]:
                key = _funnel_variant_key(result)
                if key not in selected_keys:
                    selected.append(result)
                    selected_keys.add(key)
    return selected


def _json_safe(obj):
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items() if k != "profile"}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, set):
        values = [_json_safe(v) for v in obj]
        return sorted(values, key=lambda v: json.dumps(v, ensure_ascii=False, sort_keys=True))
    if isinstance(obj, np.generic):
        return _json_safe(obj.item())
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (torch.device, torch.dtype)):
        return str(obj)
    if isinstance(obj, torch.Tensor):
        return _json_safe(obj.detach().cpu().tolist())
    return obj


def write_rvq_funnel_results(results: Sequence[Dict[str, Any]], path: str):
    if not path:
        return
    payload = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "settings": {
            "funnel_k_values": RVQ_FUNNEL_K_VALUES_ENV,
            "block_topk": int(RVQ_FUNNEL_BLOCK_TOPK),
            "final_topk_per_bpp_per_transform": int(RVQ_FUNNEL_FINAL_TOPK),
            "include_conv_stage": bool(RVQ_FUNNEL_INCLUDE_CONV_STAGE),
            "conv_topk": int(RVQ_FUNNEL_CONV_TOPK),
            "target_mode": rvq_target_mode,
            "hadamard_blocks": list(hadamard_block_candidates),
            "hadamard_signs": list(hadamard_sign_candidates),
        },
        "results": _json_safe(list(results)),
    }
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(f"RVQ funnel results written to: {out.resolve()}")


def _default_rvq_fixed_sweep_output_dir() -> Path:
    job_id = os.environ.get("SLURM_JOB_ID", "").strip()
    if job_id:
        run_name = f"rvq_fixed_sweep_{job_id}"
    else:
        run_name = f"rvq_fixed_sweep_{time.strftime('%Y%m%d_%H%M%S')}"
    return Path("runs") / run_name


def _rvq_result_summary_row(result: Dict[str, Any]) -> Dict[str, Any]:
    agg = result.get("agg") or {}
    stage_ks = result.get("stage_Ks", [])
    stage_bits = result.get("stage_bits", [])
    transform_label = result.get("transform_label") or format_transform_label(result.get("block"), result.get("with_signs"))
    row = {
        "job_id": os.environ.get("SLURM_JOB_ID", ""),
        "target_mode": rvq_target_mode,
        "include_conv": bool(rvq_include_conv),
        "profile_key": result.get("profile_key", ""),
        "profile_label": result.get("profile_label", ""),
        "reference_bpp": result.get("reference_bpp", ""),
        "index_bpp": result.get("index_bpp", ""),
        "bits": result.get("bits", ""),
        "K": result.get("K", ""),
        "stage_Ks": "x".join(str(k) for k in stage_ks),
        "stage_bits": "x".join(str(b) for b in stage_bits),
        "d": result.get("d", ""),
        "rvq_M": result.get("rvq_M", ""),
        "transform_family": result.get("transform_family", ""),
        "transform_label": transform_label,
        "block": result.get("block", ""),
        "with_signs": result.get("with_signs", ""),
        "score": result.get("score", ""),
    }
    for key in [
        "lpips",
        "clip_ii",
        "clip_ti",
        "clip_ti_fp",
        "delta_clip_ti",
        "pixel_mse",
        "pixel_mae",
        "psnr",
        "ssim",
        "hps",
        "hps_fp",
        "n",
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
    ]:
        row[key] = agg.get(key, "")
    return row


def write_rvq_fixed_sweep_results(results: Sequence[Dict[str, Any]], output_dir: str = ""):
    if not results:
        return
    out_dir = Path(output_dir).expanduser() if output_dir else _default_rvq_fixed_sweep_output_dir()
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = [_rvq_result_summary_row(r) for r in results]
    fieldnames = list(summary_rows[0].keys())
    csv_path = out_dir / "rvq_fixed_sweep_summary.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    compact_results = []
    for result, row in zip(results, summary_rows):
        compact = _json_safe(row)
        compact["agg"] = _json_safe(result.get("agg", {}))
        compact_results.append(compact)
    payload = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "settings": {
            "target_mode": rvq_target_mode,
            "include_conv": bool(rvq_include_conv),
            "stage_k_sweep": bool(RVQ_STAGE_K_SWEEP),
            "hadamard_mode": RVQ_HADAMARD_MODE_ENV,
            "active_profiles": [rvq_profile_key(p) for p in active_bpp_profiles],
        },
        "results": compact_results,
    }
    json_path = out_dir / "rvq_fixed_sweep_results.json"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(f"RVQ fixed sweep CSV written to: {csv_path.resolve()}")
    print(f"RVQ fixed sweep JSON written to: {json_path.resolve()}")


def ensure_rvq_xtx_for_layers(layer_names: Sequence[str], *, label: str):
    global rvq_xtx_stats, rvq_xtx_stats_signature_saved
    layer_names = list(dict.fromkeys(layer_names))
    if "rvq_xtx_stats" not in globals() or not isinstance(rvq_xtx_stats, dict):
        rvq_xtx_stats = {}
    missing = [name for name in layer_names if name not in rvq_xtx_stats]
    if not missing:
        print(f"{label}: reusing XTX for {len(layer_names)} layer(s).")
        return rvq_xtx_stats

    log_section(f"Collect XTX: {label}")
    print(f"{label}: collecting XTX for {len(missing)} missing layer(s).")
    log_event(
        f"missing_layers={len(missing)}, calibration_prompts={len(rvq_calibration_prompts)}, "
        f"steps={rvq_xtx_num_inference_steps}, collect_device={rvq_collect_device}, "
        f"xtx_store_device={rvq_xtx_store_device}, seed={rvq_xtx_seed}"
    )
    t0 = time.time()
    new_stats = collect_midblock_xtx_streaming_autobatch(
        pipe=pipe,
        calibration_prompts=rvq_calibration_prompts,
        layer_names=missing,
        xtx_batch_size=rvq_xtx_batch_size,
        num_inference_steps=rvq_xtx_num_inference_steps,
        guidance_scale=guidance_scale,
        collect_device=rvq_collect_device,
        xtx_store_device=rvq_xtx_store_device,
        accum_dtype=torch.float32,
        group_channels=True,
        show_progress=True,
        generator_seed=rvq_xtx_seed,
    )
    rvq_xtx_stats.update(new_stats)
    rvq_xtx_stats_signature_saved = None
    if rvq_offload_after_xtx:
        move_to_cpu_safely(pipe)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(f"{label}: XTX collected in {time.time() - t0:.1f}s")
    log_cuda_memory(f"after XTX collection for {label}")
    return rvq_xtx_stats


@torch.no_grad()
def _funnel_xtx_offdiag_ratio(layer_names: Sequence[str], *, d: int, block: int, with_signs: bool) -> float:
    total_energy = 0.0
    offdiag_energy = 0.0
    for name in layer_names:
        if name not in rvq_xtx_stats:
            continue
        layer = rvq_target_layers_all[name]
        _, meta = weight_to_vectors_lastdim(layer.weight, d=int(d))
        H = rvq_xtx_stats[name]["xtx"].detach().to(dtype=torch.float32)
        H = 0.5 * (H + H.t())
        H = transform_xtx_hadamard_blocks(
            H,
            block=int(block),
            seed=VQC.hadamard_seed + VQC.seed + stable_layer_salt(name),
            with_signs=bool(with_signs),
        )
        stats = xtx_block_energy_stats(H, meta)
        total_energy += float(stats["total"])
        offdiag_energy += float(stats["offdiag"])
    return offdiag_energy / max(total_energy, 1e-30)


def select_funnel_hadamard_blocks(
    profiles: Sequence[Dict[str, Any]],
    layer_names: Sequence[str],
    blocks: Sequence[int],
    *,
    with_signs: bool,
    topk: int,
) -> List[int]:
    blocks = [int(b) for b in blocks]
    topk = int(topk)
    if topk <= 0 or topk >= len(blocks):
        print(f"RVQ funnel block screening disabled: using all blocks {blocks}")
        return blocks

    selected: set[int] = set()
    diagnostics = []
    for d_value in sorted({int(p["d"]) for p in profiles}):
        scored = []
        for block in blocks:
            ratio = _funnel_xtx_offdiag_ratio(layer_names, d=d_value, block=block, with_signs=with_signs)
            scored.append((ratio, block))
        scored.sort(key=lambda item: item[0])
        chosen = scored[:max(1, topk)]
        selected.update(block for _, block in chosen)
        diagnostics.append((d_value, scored, chosen))

    print("RVQ funnel Hadamard block pre-screen by full-XTX offdiag ratio:")
    for d_value, scored, chosen in diagnostics:
        scored_text = ", ".join(f"H{block}:{ratio:.4f}" for ratio, block in scored)
        chosen_text = ", ".join(f"H{block}" for _, block in chosen)
        print(f"  d={d_value}: {scored_text} | selected {chosen_text}")
    selected_blocks = [block for block in blocks if block in selected]
    print(f"RVQ funnel selected Hadamard blocks: {selected_blocks}")
    return selected_blocks


def collect_rvq_layer_diagnostics(module: nn.Module, *, prefix: str = "mid_block") -> List[Dict[str, Any]]:
    diagnostics: List[Dict[str, Any]] = []
    for name, child in module.named_modules():
        if not getattr(child, "is_rvq", False):
            continue
        full_name = f"{prefix}.{name}" if prefix and name else prefix
        code_usage_stats = dict(getattr(child, "rvq_code_usage_stats", {}) or {})
        stage_distortions = [float(x) for x in getattr(child, "rvq_stage_distortions", [])]
        weight = getattr(child, "weight", None)
        weight_params = int(weight.numel()) if isinstance(weight, torch.Tensor) else int(
            getattr(child, "out_features", 0) * getattr(child, "in_features", 0)
        )
        diagnostics.append({
            "name": full_name,
            "module_type": type(child).__name__,
            "weight_params": weight_params,
            "rvq_M": int(getattr(child, "rvq_M", 0)),
            "K": int(getattr(child, "vq_K", 0)),
            "bits": int(getattr(child, "bits", 0)),
            "stage_Ks": list(getattr(child, "stage_Ks", []) or []),
            "stage_bits": list(getattr(child, "stage_bits", []) or []),
            "d": int(getattr(child, "vq_d", 0)),
            "use_hadamard": bool(getattr(child, "use_hadamard", False)),
            "activation_aware": bool(getattr(child, "activation_aware", False)),
            "stage_distortions": stage_distortions,
            "stage_distortion_sum": float(sum(stage_distortions)) if stage_distortions else float("nan"),
            "full_xtx_recode": dict(getattr(child, "rvq_full_xtx_recode_stats", {}) or {}),
            "code_usage": code_usage_stats,
        })
    return diagnostics


def summarize_rvq_diagnostics(diagnostics: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    total_weight = 0
    weighted_entropy_bpp = 0.0
    weighted_packed_bpp = 0.0
    weighted_utilization = 0.0
    weighted_norm_entropy = 0.0
    weighted_stage_distortion_sum = 0.0
    total_used = 0
    total_dead = 0
    total_entries = 0
    invalid = 0
    layers_with_stats = 0
    layers_with_distortions = 0
    for diag in diagnostics:
        weight = int(diag.get("weight_params", 0) or 0)
        if weight <= 0:
            continue
        total_weight += weight
        stage_distortions = [float(x) for x in diag.get("stage_distortions", []) if np.isfinite(float(x))]
        if stage_distortions:
            layers_with_distortions += 1
            weighted_stage_distortion_sum += weight * float(sum(stage_distortions))

        usage = diag.get("code_usage", {}) or {}
        if not usage:
            continue
        layers_with_stats += 1
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
        "layers_with_stage_distortions": int(layers_with_distortions),
        "weighted_stage_distortion_sum": float(weighted_stage_distortion_sum / denom),
    }


def run_rvq_funnel_variant(
    spec: Dict[str, Any],
    *,
    target_layer_names: Sequence[str],
    target_scope: str,
    stage: str,
    prompts: Sequence[str],
    seeds: Sequence[int],
    fp_cache: Dict[Any, Any],
):
    profile = spec["profile"]
    block = spec.get("block")
    with_signs = spec.get("with_signs")
    use_hadamard = block is not None
    cfg = make_rvq_aw_cfg(
        use_hadamard=use_hadamard,
        block=None if block is None else int(block),
        with_signs=False if with_signs is None else bool(with_signs),
        profile=profile,
    )
    transform_label = "no Hadamard" if not use_hadamard else f"Hadamard block={block}, signs={bool(with_signs)}"
    label = f"{stage} | {target_scope} | {rvq_profile_label(profile)} | {transform_label}"

    t_variant0 = time.perf_counter()
    t_quant0 = time.perf_counter()
    pipe_q, unet_q = build_rvq_midblock_pipe_from_cfg(
        cfg,
        label,
        target_layer_names=set(target_layer_names),
        xtx_stats_override=rvq_xtx_stats,
    )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t_quant = time.perf_counter() - t_quant0
    layer_diagnostics = collect_rvq_layer_diagnostics(unet_q)
    diagnostic_summary = summarize_rvq_diagnostics(layer_diagnostics)

    t_eval0 = time.perf_counter()
    if pipe_q is None:
        agg, rows = eval_with_temporary_quantized_midblock(
            pipe_obj=pipe,
            unet_q=unet_q,
            prompts=prompts,
            seeds=seeds,
            fp_cache=fp_cache,
            image_pbar=None,
            eval_label=label,
        )
    else:
        agg, rows = eval_prompts_against_fp(
            pipe,
            pipe_q,
            prompts,
            seeds,
            fp_cache=fp_cache,
            image_pbar=None,
            eval_label=label,
        )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t_eval = time.perf_counter() - t_eval0
    t_total = time.perf_counter() - t_variant0
    score = hadamard_rank_score(agg)

    result = {
        "stage": stage,
        "target_scope": target_scope,
        "target_layer_count": len(set(target_layer_names)),
        "label": label,
        "profile": dict(profile),
        "profile_key": rvq_profile_key(profile),
        "profile_label": rvq_profile_label(profile),
        "reference_bpp": float(profile.get("reference_bpp", float("nan"))),
        "index_bpp": sum(rvq_profile_stage_bits(profile)) / int(profile["d"]),
        "bits": int(max(rvq_profile_stage_bits(profile))),
        "K": max(rvq_profile_stage_ks(profile)),
        "stage_Ks": list(rvq_profile_stage_ks(profile)),
        "stage_bits": list(rvq_profile_stage_bits(profile)),
        "d": int(profile["d"]),
        "rvq_M": len(rvq_profile_stage_ks(profile)),
        "block": None if block is None else int(block),
        "with_signs": None if with_signs is None else bool(with_signs),
        "transform_family": "H" if use_hadamard else "noH",
        "score": float(score),
        "agg": agg,
        "rows": rows,
        "layer_diagnostics": layer_diagnostics,
        "code_usage_summary": diagnostic_summary,
        "timing": {"quant_build_s": float(t_quant), "eval_s": float(t_eval), "total_s": float(t_total)},
    }
    print(
        f"FUNNEL RESULT | {label} | index_bpp={result['index_bpp']:.2f} | "
        f"score={score:.6f} | quant={t_quant:.1f}s eval={t_eval:.1f}s | agg={agg}"
    )

    del pipe_q, unet_q
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    log_cuda_memory(f"after {label}")
    return result


def _make_noh_spec(profile: Dict[str, Any]) -> Dict[str, Any]:
    spec = make_sweep_spec(profile, block=hadamard_block_candidates[0], with_signs=False)
    spec["block"] = None
    spec["with_signs"] = None
    return spec


def _make_result_spec(result: Dict[str, Any]) -> Dict[str, Any]:
    profile = dict(result["profile"])
    spec = make_sweep_spec(profile, block=result["block"] or hadamard_block_candidates[0], with_signs=bool(result.get("with_signs", False)))
    spec["block"] = result.get("block")
    spec["with_signs"] = result.get("with_signs")
    return spec


def print_funnel_stage_summary(title: str, results: Sequence[Dict[str, Any]], limit: int = 10):
    print()
    print("#" * 110)
    print(title)
    for idx, result in enumerate(sorted(results, key=lambda r: float(r["score"]), reverse=True)[:limit], 1):
        block = "noH" if result.get("block") is None else f"H{result['block']}"
        agg = result["agg"]
        print(
            f"{idx:2d}. {result['stage']} | {result['target_scope']} | {result['profile_label']} | "
            f"{block:5s} | family={_funnel_transform_family(result):3s} | score={result['score']:.6f} | "
            f"LPIPS={agg['lpips']:.4f} | CLIP i-i={agg['clip_ii']:.4f} | "
            f"Ks={result.get('stage_Ks', [result['K']] * int(result['rvq_M']))} d={result['d']} M={result['rvq_M']}"
        )


def run_budgeted_funnel_sweep():
    log_section("RVQ budgeted funnel sweep")
    print(
        "RVQ_FUNNEL_SWEEP is ON: staged one-job search over d, Hadamard block, K, "
        "all-midblock Linear, and optional Conv targets."
    )

    hardcoded_linear_names = [name for name in HARDCODED_RVQ_LINEAR_TARGET_NAMES if name in rvq_target_layers_all]
    missing_hardcoded = [name for name in HARDCODED_RVQ_LINEAR_TARGET_NAMES if name not in rvq_target_layers_all]
    if missing_hardcoded:
        raise RuntimeError("Missing hardcoded RVQ Linear target(s): " + ", ".join(missing_hardcoded))
    all_linear_names = list(linear_target_layers_all.keys())
    conv_names = [name for name in HARDCODED_RVQ_CONV_TARGET_NAMES if name in rvq_target_layers_all]

    ensure_rvq_xtx_for_layers(hardcoded_linear_names, label="funnel stage A hardcoded Linear")

    base_profiles = _build_rvq_bpp_profiles([256], vector_dims=rvq_vector_dims)
    funnel_blocks = select_funnel_hadamard_blocks(
        base_profiles,
        hardcoded_linear_names,
        hadamard_block_candidates,
        with_signs=False,
        topk=RVQ_FUNNEL_BLOCK_TOPK,
    )
    base_specs = [
        make_sweep_spec(profile, block, False)
        for block in funnel_blocks
        for profile in base_profiles
    ]

    rvq_funnel_results: List[Dict[str, Any]] = []
    for spec in base_specs:
        rvq_funnel_results.append(run_rvq_funnel_variant(
            spec,
            target_layer_names=hardcoded_linear_names,
            target_scope="hardcoded_linear",
            stage="A:K256-d-block",
            prompts=hadamard_sweep_prompts,
            seeds=hadamard_sweep_seeds,
            fp_cache=rvq_fp_image_cache,
        ))
    print_funnel_stage_summary("FUNNEL STAGE A SUMMARY: hardcoded Linear, K=256", rvq_funnel_results)

    best_hadamard_by_profile: Dict[str, Dict[str, Any]] = {}
    for profile in base_profiles:
        rows = [r for r in rvq_funnel_results if result_matches_profile(r, profile) and r.get("block") is not None]
        if rows:
            best_hadamard_by_profile[rvq_profile_key(profile)] = max(rows, key=lambda r: float(r["score"]))

    stage_b_results = []
    for profile in base_profiles:
        result = run_rvq_funnel_variant(
            _make_noh_spec(profile),
            target_layer_names=hardcoded_linear_names,
            target_scope="hardcoded_linear",
            stage="B:noH",
            prompts=rvq_final_comparison_prompts,
            seeds=rvq_final_comparison_seeds,
            fp_cache=rvq_fp_image_cache,
        )
        stage_b_results.append(result)
        rvq_funnel_results.append(result)
    print_funnel_stage_summary("FUNNEL STAGE B SUMMARY: no-Hadamard vs best Hadamard", stage_b_results)

    k_values = _parse_rvq_k_values(RVQ_FUNNEL_K_VALUES_ENV)
    k_stage_results = []
    for target_tuple in [t for t in rvq_reference_bpp_targets if t[0] in {"3.15", "2.15"}]:
        target_key = target_tuple[0]
        for family in ("H", "noH"):
            family_candidates = [
                r for r in rvq_funnel_results
                if r["target_scope"] == "hardcoded_linear"
                and _profile_target_key(r["profile"]) == target_key
                and _funnel_transform_family(r) == family
            ]
            if not family_candidates:
                continue
            anchor = max(family_candidates, key=lambda r: float(r["score"]))
            fixed_d = int(anchor["d"])
            fixed_block = anchor.get("block")
            fixed_with_signs = anchor.get("with_signs")
            for K_value in k_values:
                if int(K_value) == 256:
                    continue
                profile = _build_rvq_bpp_profiles([int(K_value)], vector_dims=[fixed_d], reference_targets=[target_tuple])[0]
                if fixed_block is None:
                    spec = _make_noh_spec(profile)
                else:
                    spec = make_sweep_spec(profile, int(fixed_block), bool(fixed_with_signs))
                result = run_rvq_funnel_variant(
                    spec,
                    target_layer_names=hardcoded_linear_names,
                    target_scope="hardcoded_linear",
                    stage=f"C:K-ablation-{family}",
                    prompts=hadamard_sweep_prompts,
                    seeds=hadamard_sweep_seeds,
                    fp_cache=rvq_fp_image_cache,
                )
                k_stage_results.append(result)
                rvq_funnel_results.append(result)
    if k_stage_results:
        print_funnel_stage_summary("FUNNEL STAGE C SUMMARY: K ablation on selected d/block for 3/2 bpp", k_stage_results)

    final_pool = [
        r for r in rvq_funnel_results
        if r["target_scope"] == "hardcoded_linear" and _profile_target_key(r["profile"]) in {"4.15", "3.15", "2.15"}
    ]
    final_candidates = select_funnel_final_candidates(final_pool, RVQ_FUNNEL_FINAL_TOPK)
    print_funnel_stage_summary(
        "FUNNEL SELECTED FINAL CANDIDATES FOR ALL-LINEAR: top-k per bpp and per H/noH",
        final_candidates,
        limit=len(final_candidates),
    )
    ensure_rvq_xtx_for_layers(all_linear_names, label="funnel stage D all midblock Linear")

    all_linear_results = []
    for candidate in final_candidates:
        result = run_rvq_funnel_variant(
            _make_result_spec(candidate),
            target_layer_names=all_linear_names,
            target_scope="all_midblock_linear",
            stage="D:all-linear",
            prompts=rvq_final_comparison_prompts,
            seeds=rvq_final_comparison_seeds,
            fp_cache=rvq_fp_image_cache,
        )
        all_linear_results.append(result)
        rvq_funnel_results.append(result)
    print_funnel_stage_summary("FUNNEL STAGE D SUMMARY: all mid_block Linear for top candidates", all_linear_results)

    if RVQ_FUNNEL_INCLUDE_CONV_STAGE and conv_names and all_linear_results:
        conv_target_names = list(dict.fromkeys(all_linear_names + conv_names))
        ensure_rvq_xtx_for_layers(conv_target_names, label="funnel stage E all Linear + hardcoded Conv")
        conv_candidates = select_funnel_final_candidates(all_linear_results, RVQ_FUNNEL_CONV_TOPK)
        conv_results = []
        for candidate in conv_candidates:
            result = run_rvq_funnel_variant(
                _make_result_spec(candidate),
                target_layer_names=conv_target_names,
                target_scope="all_linear_plus_hardcoded_conv",
                stage="E:conv",
                prompts=rvq_final_comparison_prompts,
                seeds=rvq_final_comparison_seeds,
                fp_cache=rvq_fp_image_cache,
            )
            conv_results.append(result)
            rvq_funnel_results.append(result)
        print_funnel_stage_summary("FUNNEL STAGE E SUMMARY: add Conv targets for top Linear candidate(s)", conv_results)
    elif RVQ_FUNNEL_INCLUDE_CONV_STAGE:
        print("FUNNEL STAGE E skipped: no Conv targets or no valid all-Linear result.")

    print_funnel_stage_summary("FINAL RVQ FUNNEL SUMMARY", rvq_funnel_results, limit=20)
    write_rvq_funnel_results(rvq_funnel_results, RVQ_FUNNEL_RESULTS_JSON)
    return rvq_funnel_results


# %% cell 38
# === Hadamard/bpp sweep part 1/2 ===

if rvq_budgeted_funnel:
    rvq_funnel_results = run_budgeted_funnel_sweep()
else:
    log_section("Hadamard/bpp sweep part 1/2")
    hadamard_sweep_results = []
    hadamard_sweep_completed_keys = set()

    run_hadamard_sweep_chunk(hadamard_sweep_specs_part1, "RVQ sweep 1/2")

    print()
    print("#"*110)
    print(f"Sweep part 1/2 done: {len(hadamard_sweep_results)}/{len(sweep_specs)} variants")
    for r in sorted(hadamard_sweep_results, key=lambda r: r["score"], reverse=True):
        transform_label = r.get("transform_label") or format_transform_label(r.get("block"), r.get("with_signs"))
        print(
            f"{r['profile_label']:28s} | {transform_label}, "
            f"score={r['score']:.6f}, LPIPS={r['agg']['lpips']:.4f}, CLIP i-i={r['agg']['clip_ii']:.4f}"
        )

# %% cell 39
# === Hadamard/bpp sweep part 2/2 and best-by-bpp ===

if not rvq_budgeted_funnel:
    log_section("Hadamard/bpp sweep part 2/2")
    if "hadamard_sweep_results" not in globals():
        hadamard_sweep_results = []
    if "hadamard_sweep_completed_keys" not in globals():
        hadamard_sweep_completed_keys = completed_sweep_keys(hadamard_sweep_results)

    run_hadamard_sweep_chunk(hadamard_sweep_specs_part2, "RVQ sweep 2/2")

    if len(hadamard_sweep_results) < len(sweep_specs):
        done = completed_sweep_keys(hadamard_sweep_results)
        missing = [spec for spec in sweep_specs if sweep_spec_key(spec) not in done]
        raise RuntimeError(f"Sweep is incomplete: run missing variants first: {missing}")

    hadamard_best_by_profile = {}
    for profile in active_bpp_profiles:
        key = rvq_profile_key(profile)
        rows = [r for r in hadamard_sweep_results if result_matches_profile(r, profile)]
        if not rows:
            raise RuntimeError(f"No sweep results for profile {key}")
        hadamard_best_by_profile[key] = max(rows, key=lambda r: r["score"])

    hadamard_top3 = sorted(hadamard_sweep_results, key=lambda r: r["score"], reverse=True)[:3]

    print()
    print("#"*110)
    print(f"Sweep part 2/2 done: {len(hadamard_sweep_results)}/{len(sweep_specs)} variants")
    print("BEST transform variant inside each bpp profile")
    for profile in active_bpp_profiles:
        r = hadamard_best_by_profile[rvq_profile_key(profile)]
        transform_label = r.get("transform_label") or format_transform_label(r.get("block"), r.get("with_signs"))
        print(
            f"{r['profile_label']:28s} | {transform_label}, "
            f"score={r['score']:.6f}, LPIPS={r['agg']['lpips']:.4f}, CLIP i-i={r['agg']['clip_ii']:.4f}"
        )

    print()
    print("GLOBAL TOP-3 Hadamard variants by score = CLIP image-image - LPIPS")
    for i, r in enumerate(hadamard_top3, 1):
        transform_label = r.get("transform_label") or format_transform_label(r.get("block"), r.get("with_signs"))
        print(
            f"{i}. {r['profile_label']} | {transform_label}, "
            f"score={r['score']:.6f}, agg={r['agg']}"
        )

# %% cell 40
# === Final comparison: RVQ noH vs best Hadamard per bpp ===

if not rvq_budgeted_funnel and rvq_fixed_transform_sweep:
    rvq_hadamard_comparison = hadamard_sweep_results
    print()
    print("#"*110)
    print(f"FINAL SUMMARY: fixed transform sweep ({rvq_hadamard_mode})")
    for r in sorted(rvq_hadamard_comparison, key=lambda item: (float(item.get("reference_bpp", 0.0)), item["profile_key"])):
        a = r["agg"]
        transform_label = r.get("transform_label") or format_transform_label(r.get("block"), r.get("with_signs"))
        print(
            f"{r['profile_label']:36s} | {transform_label:12s} | index_bpp={r['index_bpp']:.3f} | "
            f"score={r['score']:.6f} | LPIPS={a['lpips']:.4f} | CLIP i-i={a['clip_ii']:.4f} | "
            f"CLIP t-i={a['clip_ti']:.4f}"
        )
    write_rvq_fixed_sweep_results(rvq_hadamard_comparison, RVQ_FIXED_SWEEP_OUTPUT_DIR)

if not rvq_budgeted_funnel and not rvq_fixed_transform_sweep:
    log_section("Final comparison")
    if "hadamard_best_by_profile" not in globals():
        raise RuntimeError("Run both Hadamard/bpp sweep cells first to create hadamard_best_by_profile")
    if "rvq_xtx_stats" not in globals():
        raise RuntimeError("Run the setup cell first to collect rvq_xtx_stats")
    if "rvq_fp_image_cache" not in globals():
        raise RuntimeError("Run the setup cell first to precompute rvq_fp_image_cache")
    if not rvq_fast_smoke_test and len(hadamard_sweep_results) < len(sweep_specs):
        raise RuntimeError("Run both Hadamard/bpp sweep cells before the final comparison")

    if rvq_fast_smoke_test:
        print("RVQ smoke mode: final comparison skipped. Sweep results are in hadamard_sweep_results.")
        rvq_hadamard_comparison = hadamard_sweep_results
    else:
        rvq_comparison_prompts = list(rvq_final_comparison_prompts)
        rvq_comparison_seeds = list(rvq_final_comparison_seeds)

        same_eval_grid = (
            list(rvq_comparison_prompts) == list(hadamard_sweep_prompts)
            and list(rvq_comparison_seeds) == list(hadamard_sweep_seeds)
        )
        if not same_eval_grid:
            raise RuntimeError(
                "Final prompts/seeds differ from the Hadamard sweep. "
                "Reusing sweep Hadamard results would be invalid; set final prompts/seeds equal to sweep or rerun final quantization explicitly."
            )

        print(
            f"Final RVQ comparison setup: no-Hadamard will be quantized once per bpp profile; "
            f"best Hadamard per profile is reused from sweep | "
            f"profiles={len(active_bpp_profiles)} | "
            f"eval_pairs={len(rvq_comparison_prompts) * len(rvq_comparison_seeds)} | "
            f"eval_steps={num_inference_steps}"
        )

        reused_hadamard_results = []
        for profile in active_bpp_profiles:
            key = rvq_profile_key(profile)
            src = hadamard_best_by_profile[key]
            reused_hadamard_results.append({
                "label": src["label"],
                "profile_key": src["profile_key"],
                "profile_label": src["profile_label"],
                "reference_bpp": src["reference_bpp"],
                "index_bpp": src["index_bpp"],
                "bits": src["bits"],
                "K": src.get("K", 1 << int(src["bits"])),
                "stage_Ks": src.get("stage_Ks", []),
                "stage_bits": src.get("stage_bits", []),
                "d": src["d"],
                "rvq_M": src["rvq_M"],
                "block": src["block"],
                "with_signs": src["with_signs"],
                "score": src["score"],
                "agg": src["agg"],
                "rows": src.get("rows", []),
                "method": "hadamard",
                "reused_from_sweep": True,
            })

        final_missing_fp = count_missing_fp_images(
            rvq_fp_image_cache,
            rvq_comparison_prompts,
            rvq_comparison_seeds,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
        )
        final_image_total = final_missing_fp + len(active_bpp_profiles) * len(rvq_comparison_prompts) * len(rvq_comparison_seeds)

        no_h_results = []
        with tqdm(total=final_image_total, desc="Final RVQ noH generated images", unit="image") as image_pbar:
            if final_missing_fp:
                precompute_fp_image_cache(
                    pipe,
                    rvq_comparison_prompts,
                    rvq_comparison_seeds,
                    cache=rvq_fp_image_cache,
                    image_pbar=image_pbar,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                )

            for profile in active_bpp_profiles:
                label = f"RVQ-AW {rvq_profile_label(profile)} no Hadamard"
                no_h_cfg = make_rvq_aw_cfg(use_hadamard=False, profile=profile)
                t_variant0 = time.perf_counter()
                t_quant0 = time.perf_counter()
                pipe_q, unet_q = build_rvq_midblock_pipe_from_cfg(no_h_cfg, label)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                t_quant = time.perf_counter() - t_quant0

                t_eval0 = time.perf_counter()
                if pipe_q is None:
                    agg, rows = eval_with_temporary_quantized_midblock(
                        pipe_obj=pipe,
                        unet_q=unet_q,
                        prompts=rvq_comparison_prompts,
                        seeds=rvq_comparison_seeds,
                        fp_cache=rvq_fp_image_cache,
                        image_pbar=image_pbar,
                        eval_label=label,
                    )
                else:
                    agg, rows = eval_prompts_against_fp(
                        pipe,
                        pipe_q,
                        rvq_comparison_prompts,
                        rvq_comparison_seeds,
                        fp_cache=rvq_fp_image_cache,
                        image_pbar=image_pbar,
                        eval_label=label,
                    )
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                t_eval = time.perf_counter() - t_eval0
                log_event(f"Variant timing | {label} | quant_build={t_quant:.1f}s, eval={t_eval:.1f}s, total={time.perf_counter() - t_variant0:.1f}s")
                score = hadamard_rank_score(agg)

                no_h_result = {
                    "label": label,
                    "profile_key": rvq_profile_key(profile),
                    "profile_label": rvq_profile_label(profile),
                    "reference_bpp": float(profile.get("reference_bpp", float("nan"))),
                    "index_bpp": sum(rvq_profile_stage_bits(profile)) / int(profile["d"]),
                    "bits": int(max(rvq_profile_stage_bits(profile))),
                    "K": rvq_profile_K(profile),
                    "stage_Ks": list(rvq_profile_stage_ks(profile)),
                    "stage_bits": list(rvq_profile_stage_bits(profile)),
                    "d": int(profile["d"]),
                    "rvq_M": len(rvq_profile_stage_ks(profile)),
                    "block": None,
                    "with_signs": None,
                    "score": score,
                    "agg": agg,
                    "rows": rows,
                    "method": "no_hadamard",
                    "reused_from_sweep": False,
                }
                no_h_results.append(no_h_result)

                print()
                print("#"*110)
                print(f"FINAL RESULT | {label}")
                print(f"score={score:.6f}")
                print(agg)

                del pipe_q, unet_q
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        rvq_hadamard_comparison = []
        for profile in active_bpp_profiles:
            key = rvq_profile_key(profile)
            rvq_hadamard_comparison.append(next(r for r in no_h_results if r["profile_key"] == key))
            rvq_hadamard_comparison.append(next(r for r in reused_hadamard_results if r["profile_key"] == key))

        print()
        print("#"*110)
        print("FINAL SUMMARY: RVQ noH vs best Hadamard per bpp")
        for profile in active_bpp_profiles:
            key = rvq_profile_key(profile)
            no_h = next(r for r in no_h_results if r["profile_key"] == key)
            had = next(r for r in reused_hadamard_results if r["profile_key"] == key)
            delta = had["score"] - no_h["score"]
            print()
            print(f"{rvq_profile_label(profile)} | reference_bpp~{profile['reference_bpp']} | index_bpp={had['index_bpp']:.2f}")
            for r in (no_h, had):
                a = r["agg"]
                source = "sweep" if r.get("reused_from_sweep", False) else "final"
                block = "noH" if r["block"] is None else f"H{r['block']}"
                print(
                    f"  {block:5s} | source={source:5s} | "
                    f"score={r['score']:.6f} | "
                    f"LPIPS={a['lpips']:.4f} | "
                    f"CLIP i-i={a['clip_ii']:.4f} | "
                    f"CLIP t-i={a['clip_ti']:.4f} | "
                    f"HPS={a['hps']:.4f} | "
                    f"HPS FP={a['hps_fp']:.4f}"
                )
            print(f"  Delta Hadamard - noH: {delta:+.6f}")
