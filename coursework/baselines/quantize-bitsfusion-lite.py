# -*- coding: utf-8 -*-
"""BitsFusion-lite mixed-bit baseline for SD1.5.

This is a deliberately practical, no-gradient approximation of the useful
BitsFusion idea for our setting: collect timestep-aware calibration inputs,
estimate layer reconstruction error for several scalar bit-widths, then greedily
allocate per-layer weight bits under a requested average bpp budget. For each
chosen layer bit-width, the actual scalar quantizer is the PTQ4DM-like
per-output-channel clipping-grid quantizer used by the other baselines.

This is not the full BitsFusion training/distillation recipe. It is the
training-free mixed-bit scalar baseline we can compare directly to RVQ/VQDM
codebook-only runs.

The script mirrors the RVQ/VQDM evaluation protocol: PickScore validation
prompts, DDIM, fixed seeds, CLIP image-image/text-image, LPIPS, cheap pixel
metrics, forward drift, and per-variant timing.
"""

from __future__ import annotations

import atexit
import builtins
import copy
import csv
import json
import logging
import math
import os
import random
import subprocess
import sys
import time
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

os.environ.setdefault("PYTHONNOUSERSITE", "1")


def _pre_torch_env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None or value == "":
        return bool(default)
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _pre_torch_env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None or value == "":
        return int(default)
    return int(value)


def _query_nvidia_smi_gpu_state() -> Dict[str, Tuple[int, int]]:
    query = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,utilization.gpu,memory.used",
            "--format=csv,noheader,nounits",
        ],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if query.returncode != 0:
        raise RuntimeError(query.stderr.strip() or "nvidia-smi query failed")

    state: Dict[str, Tuple[int, int]] = {}
    for line in query.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 3:
            continue
        idx, util, mem = parts[:3]
        try:
            state[str(int(idx))] = (int(util), int(mem))
        except ValueError:
            continue
    return state


def _preselect_idle_gpus_before_torch_import():
    if not _pre_torch_env_bool("BITSFUSION_SELECT_IDLE_GPUS", True):
        print("BITSFUSION_SELECT_IDLE_GPUS=0; keeping caller GPU visibility.")
        return

    try:
        gpu_state = _query_nvidia_smi_gpu_state()
    except FileNotFoundError:
        message = "nvidia-smi not found; cannot verify idle GPUs before torch import."
        if _pre_torch_env_bool("BITSFUSION_REQUIRE_IDLE_GPUS", True):
            raise SystemExit(message)
        print("WARNING: " + message)
        return
    except Exception as exc:
        message = f"nvidia-smi idle-GPU preselection failed: {type(exc).__name__}: {exc}"
        if _pre_torch_env_bool("BITSFUSION_REQUIRE_IDLE_GPUS", True):
            raise SystemExit(message)
        print("WARNING: " + message)
        return

    if not gpu_state:
        message = "nvidia-smi returned no GPU rows; cannot verify idle GPUs."
        if _pre_torch_env_bool("BITSFUSION_REQUIRE_IDLE_GPUS", True):
            raise SystemExit(message)
        print("WARNING: " + message)
        return

    max_used_mem_mib = _pre_torch_env_int("BITSFUSION_IDLE_GPU_MAX_USED_MEM_MIB", 2048)
    max_devices = _pre_torch_env_int("BITSFUSION_AUTO_MAX_DEVICES", 0)  # 0 = all idle GPUs
    require_idle = _pre_torch_env_bool("BITSFUSION_REQUIRE_IDLE_GPUS", True)
    validate_preset = _pre_torch_env_bool("BITSFUSION_VALIDATE_PRESET_GPUS", True)
    preset = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()

    def is_idle(idx: str) -> bool:
        util, mem = gpu_state[idx]
        return util == 0 and mem <= max_used_mem_mib

    if preset:
        preset_ids = [part.strip() for part in preset.split(",") if part.strip()]
        integer_ids = [part for part in preset_ids if part.isdigit()]
        if validate_preset and integer_ids:
            busy = [idx for idx in integer_ids if idx in gpu_state and not is_idle(idx)]
            missing = [idx for idx in integer_ids if idx not in gpu_state]
            if busy or missing:
                details = []
                for idx in busy:
                    util, mem = gpu_state[idx]
                    details.append(f"gpu{idx}: util={util}%, mem={mem}MiB")
                for idx in missing:
                    details.append(f"gpu{idx}: not found by nvidia-smi")
                if require_idle:
                    raise SystemExit(
                        "CUDA_VISIBLE_DEVICES contains non-idle/unavailable GPU(s); refusing to start: "
                        + "; ".join(details)
                    )
                print("WARNING: preset CUDA_VISIBLE_DEVICES has non-idle GPU(s): " + "; ".join(details))
        print(f"CUDA_VISIBLE_DEVICES is preset; using caller-selected GPUs: {preset}")
        return

    idle_gpus = [idx for idx in sorted(gpu_state, key=lambda x: int(x)) if is_idle(idx)]
    if max_devices > 0:
        idle_gpus = idle_gpus[:max_devices]

    if not idle_gpus:
        snapshot = ", ".join(
            f"gpu{idx}: util={util}%, mem={mem}MiB" for idx, (util, mem) in sorted(gpu_state.items(), key=lambda x: int(x[0]))
        )
        message = (
            f"No idle GPUs found with 0% utilization and <= {max_used_mem_mib} MiB used memory. "
            f"GPU snapshot: {snapshot}"
        )
        if require_idle:
            raise SystemExit(message)
        print("WARNING: " + message)
        return

    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(idle_gpus)
    print(
        "Selected idle GPUs before torch import: "
        f"{os.environ['CUDA_VISIBLE_DEVICES']} "
        f"(0% utilization, <= {max_used_mem_mib} MiB used memory)"
    )


_preselect_idle_gpus_before_torch_import()

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from diffusers import DDIMScheduler, StableDiffusionPipeline
from tqdm.auto import tqdm

import lpips
import open_clip


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
LOGGER = logging.getLogger("bitsfusion-lite")


def _format_seconds(seconds: float) -> str:
    seconds_i = int(seconds)
    h, rem = divmod(seconds_i, 3600)
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


atexit.register(_finish_logging)
sys.excepthook = _log_uncaught_exception
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None or value == "":
        return bool(default)
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None or value == "":
        return int(default)
    return int(value)


def _env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    if value is None or value == "":
        return float(default)
    return float(value)


def _parse_int_list(value: str, *, min_value: int = 1) -> List[int]:
    result: List[int] = []
    for part in str(value).replace(";", ",").split(","):
        part = part.strip()
        if not part:
            continue
        item = int(part)
        if item < min_value:
            raise ValueError(f"Expected values >= {min_value}, got {item}")
        if item not in result:
            result.append(item)
    if not result:
        raise ValueError("Expected at least one integer value")
    return result


def _parse_float_list(value: str, *, min_value: float = 0.0) -> List[float]:
    result: List[float] = []
    for part in str(value).replace(";", ",").split(","):
        part = part.strip()
        if not part:
            continue
        item = float(part)
        if item < min_value:
            raise ValueError(f"Expected values >= {min_value}, got {item}")
        if item not in result:
            result.append(item)
    if not result:
        raise ValueError("Expected at least one float value")
    return result


def _parse_float_sequence(value: str, *, min_value: float = 0.0) -> List[float]:
    result: List[float] = []
    for part in str(value).replace(";", ",").split(","):
        part = part.strip()
        if not part:
            continue
        item = float(part)
        if item < min_value:
            raise ValueError(f"Expected values >= {min_value}, got {item}")
        result.append(item)
    if not result:
        raise ValueError("Expected at least one float value")
    return result


def _parse_str_list(value: str) -> List[str]:
    result: List[str] = []
    for part in str(value).replace(";", ",").split(","):
        part = part.strip().lower()
        if part and part not in result:
            result.append(part)
    return result


def set_global_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _cuda_mem_info(device: Optional[torch.device | str | int] = None) -> Optional[Dict[str, float]]:
    if not torch.cuda.is_available():
        return None
    dev = None
    if device is not None:
        dev = torch.device(f"cuda:{device}") if isinstance(device, int) else torch.device(device)
        if dev.type != "cuda":
            return None
        if dev.index is None:
            dev = torch.device(f"cuda:{torch.cuda.current_device()}")
    free, total = torch.cuda.mem_get_info(dev)
    return {
        "free": float(free),
        "total": float(total),
        "free_gib": float(free) / 1024**3,
        "total_gib": float(total) / 1024**3,
    }


def _cuda_device_summaries() -> List[Dict[str, Any]]:
    if not torch.cuda.is_available():
        return []
    summaries: List[Dict[str, Any]] = []
    for idx in range(torch.cuda.device_count()):
        try:
            props = torch.cuda.get_device_properties(idx)
            free, total = torch.cuda.mem_get_info(torch.device(f"cuda:{idx}"))
            summaries.append(
                {
                    "idx": idx,
                    "name": props.name,
                    "total_gib": float(total) / 1024**3,
                    "free_gib": float(free) / 1024**3,
                }
            )
        except Exception:
            summaries.append({"idx": idx, "name": "unknown", "total_gib": 0.0, "free_gib": 0.0})
    return summaries


def log_cuda_memory(label: str = ""):
    if not torch.cuda.is_available():
        return
    try:
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


def _is_cuda_oom(exc: BaseException) -> bool:
    msg = str(exc).lower()
    return isinstance(exc, torch.cuda.OutOfMemoryError) or (
        isinstance(exc, RuntimeError)
        and ("out of memory" in msg or "cuda error: out of memory" in msg)
    )


def _clear_cuda():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass


def synchronize_cuda(device: Optional[str | torch.device] = None):
    if not torch.cuda.is_available():
        return
    try:
        if device is not None:
            dev = torch.device(device)
            if dev.type == "cuda":
                torch.cuda.synchronize(dev)
            return
        for idx in range(torch.cuda.device_count()):
            torch.cuda.synchronize(torch.device(f"cuda:{idx}"))
    except Exception:
        torch.cuda.synchronize()


def _module_device(module: nn.Module) -> torch.device:
    return next(module.parameters()).device


def force_pipeline_device_dtype(pipe, device: str | torch.device, dtype: torch.dtype):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pipe = pipe.to(device, torch_dtype=dtype)
    for component_name in ("unet", "vae", "text_encoder", "safety_checker"):
        component = getattr(pipe, component_name, None)
        if isinstance(component, nn.Module):
            component.to(device=device, dtype=dtype)
    return pipe


def optimize_pipeline_for_inference(pipe_obj, label: str):
    try:
        pipe_obj.set_progress_bar_config(disable=True)
    except Exception:
        pass
    for component_name in ("unet", "vae", "text_encoder"):
        component = getattr(pipe_obj, component_name, None)
        if isinstance(component, nn.Module):
            component.eval()
    if torch.cuda.is_available():
        try:
            pipe_obj.unet.to(memory_format=torch.channels_last)
            pipe_obj.vae.to(memory_format=torch.channels_last)
            log_event(f"Inference optimization applied to {label}: channels_last")
        except Exception as exc:
            log_event(f"Inference optimization skipped for {label}: {type(exc).__name__}: {exc}")
    return pipe_obj


def _json_default(obj):
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, torch.dtype):
        return str(obj)
    if isinstance(obj, torch.device):
        return str(obj)
    return str(obj)


def _add_file_logger(output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(output_dir / "bitsfusion_lite.log", encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s", "%H:%M:%S"))
    LOGGER.addHandler(file_handler)


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
RUN_STAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_ROOT = Path(os.environ.get("BITSFUSION_OUTPUT_ROOT", str(SCRIPT_DIR / "runs"))).expanduser()
OUTPUT_DIR = OUTPUT_ROOT / os.environ.get("BITSFUSION_RUN_NAME", f"bitsfusion_lite_{RUN_STAMP}")
_add_file_logger(OUTPUT_DIR)


@dataclass(frozen=True)
class BitsFusionLiteConfig:
    target_bpp: float
    allowed_bits: Tuple[int, ...]
    mode: str = "mixed_weight_only"
    weight_bits: int = 0
    activation_bits: int = 0
    weight_scale_bits: int = 16
    activation_scheme: str = "dynamic_per_sample"

    @property
    def label(self) -> str:
        target = f"{float(self.target_bpp):.2f}".replace(".", "p")
        bit_label = "x".join(str(int(x)) for x in self.allowed_bits)
        if self.activation_bits > 0:
            return f"bitsfusion_lite_t{target}bpp_bits{bit_label}_a{self.activation_bits}"
        return f"bitsfusion_lite_t{target}bpp_bits{bit_label}"

    def with_layer_bits(self, weight_bits: int) -> "BitsFusionLiteConfig":
        return BitsFusionLiteConfig(
            target_bpp=float(self.target_bpp),
            allowed_bits=tuple(int(x) for x in self.allowed_bits),
            mode=self.mode,
            weight_bits=int(weight_bits),
            activation_bits=int(self.activation_bits),
            weight_scale_bits=int(self.weight_scale_bits),
            activation_scheme=str(self.activation_scheme),
        )


HARDCODED_LINEAR_TARGET_NAMES = [
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

HARDCODED_CONV_TARGET_NAMES = [
    "mid_block.attentions.0.proj_in",
    "mid_block.attentions.0.proj_out",
    "mid_block.resnets.0.conv1",
    "mid_block.resnets.0.conv2",
    "mid_block.resnets.1.conv1",
    "mid_block.resnets.1.conv2",
]


GLOBAL_SEED = _env_int("BITSFUSION_SEED", 20260515)
set_global_seed(GLOBAL_SEED)

torch.backends.cudnn.allow_tf32 = _env_bool("BITSFUSION_ALLOW_TF32", True)
torch.backends.cuda.matmul.allow_tf32 = _env_bool("BITSFUSION_ALLOW_TF32", True)

RUNTIME_DEVICE = os.environ.get("BITSFUSION_RUNTIME_DEVICE") or ("cuda:0" if torch.cuda.is_available() else "cpu")
RUNTIME_DTYPE = torch.float16 if str(RUNTIME_DEVICE).startswith("cuda") else torch.float32
QUANT_DEVICE = os.environ.get("BITSFUSION_QUANT_DEVICE") or RUNTIME_DEVICE
METRIC_DEVICE = os.environ.get("BITSFUSION_METRIC_DEVICE")
if not METRIC_DEVICE:
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        METRIC_DEVICE = f"cuda:{torch.cuda.device_count() - 1}"
    else:
        METRIC_DEVICE = RUNTIME_DEVICE

MODEL_ID = os.environ.get("BITSFUSION_MODEL_ID", "runwayml/stable-diffusion-v1-5")
NUM_INFERENCE_STEPS = _env_int("BITSFUSION_STEPS", 50)
GUIDANCE_SCALE = _env_float("BITSFUSION_GUIDANCE", 5.0)
EVAL_PROMPT_COUNT = _env_int("BITSFUSION_EVAL_PROMPTS", 32)
EVAL_SEEDS = _parse_int_list(os.environ.get("BITSFUSION_EVAL_SEEDS", "1234,4321,999"), min_value=0)
TARGET_BPPS = _parse_float_list(os.environ.get("BITSFUSION_TARGET_BPPS", "4.15,3.15,2.15,1.99"), min_value=0.1)
LAYER_BITS_LIST = sorted(set(_parse_int_list(os.environ.get("BITSFUSION_LAYER_BITS", "1,2,3,4"), min_value=1)))
ACTIVATION_BITS = _env_int("BITSFUSION_ACT_BITS", 0)
ACTIVATION_SCHEME = os.environ.get("BITSFUSION_ACT_SCHEME", "dynamic_per_sample").strip().lower()
TARGET_MODE = os.environ.get("BITSFUSION_TARGET_MODE", "all_midblock").strip().lower()
SINGLE_TARGET_LAYER = os.environ.get(
    "BITSFUSION_SINGLE_TARGET_LAYER",
    "mid_block.attentions.0.transformer_blocks.0.ff.net.0.proj",
)
INCLUDE_CONV = _env_bool("BITSFUSION_INCLUDE_CONV", True)
GEN_BATCH_ENV = _env_int("BITSFUSION_GEN_BATCH", 0)
GEN_BATCH_MAX = _env_int("BITSFUSION_GEN_BATCH_MAX", 0)
GEN_BATCH_MIN = _env_int("BITSFUSION_GEN_BATCH_MIN", 1)
GPU_TARGET_MEM_FRACTION = _env_float("BITSFUSION_GPU_TARGET_MEM_FRACTION", 0.78)
GPU_RESERVE_GIB = _env_float("BITSFUSION_GPU_RESERVE_GIB", 4.0)
GEN_ESTIMATED_GIB_PER_IMAGE = _env_float("BITSFUSION_GEN_ESTIMATED_GIB_PER_IMAGE", 1.8)
COMPUTE_FORWARD_METRICS = _env_bool("BITSFUSION_FORWARD_METRICS", True)
FORWARD_METRIC_PROMPTS = _env_int("BITSFUSION_FORWARD_METRIC_PROMPTS", 2)
FORWARD_METRIC_TIMESTEPS = _env_int("BITSFUSION_FORWARD_METRIC_TIMESTEPS", 3)
FORWARD_METRIC_SEED = _env_int("BITSFUSION_FORWARD_METRIC_SEED", 20260515)
CALIB_PROMPT_COUNT = _env_int("BITSFUSION_CALIB_PROMPTS", 16)
CALIB_SEEDS = _parse_int_list(os.environ.get("BITSFUSION_CALIB_SEEDS", "777"), min_value=0)
CALIB_STEPS = _env_int("BITSFUSION_CALIB_STEPS", NUM_INFERENCE_STEPS)
CALIB_MAX_ROWS_PER_LAYER = _env_int("BITSFUSION_CALIB_MAX_ROWS_PER_LAYER", 4096)
CALIB_GEN_BATCH_ENV = _env_int("BITSFUSION_CALIB_BATCH", 0)
CLIP_RATIO_GRID = _parse_float_list(
    os.environ.get("BITSFUSION_CLIP_RATIO_GRID", "0.4,0.5,0.6,0.7,0.8,0.9,0.95,1.0"),
    min_value=1e-6,
)
if 1.0 not in CLIP_RATIO_GRID:
    CLIP_RATIO_GRID = sorted(CLIP_RATIO_GRID + [1.0])
if any(float(x) > 1.0 for x in CLIP_RATIO_GRID):
    raise ValueError("BITSFUSION_CLIP_RATIO_GRID must contain clipping ratios <= 1.0; use QDiff-lite for range expansion.")
ALLOCATION_POLICY = os.environ.get("BITSFUSION_ALLOCATION_POLICY", "semantic_sensitivity_greedy").strip().lower()
BALANCED_INTEGER = _env_bool("BITSFUSION_BALANCED_INTEGER", True)
SEMANTIC_PROTECT = _env_bool("BITSFUSION_SEMANTIC_PROTECT", True)
SEMANTIC_MIN_BITS = _env_int("BITSFUSION_SEMANTIC_MIN_BITS", 2)
CROSS_ATTN_SENSITIVITY = _env_float("BITSFUSION_CROSS_ATTN_SENSITIVITY", 1.50)
SELF_ATTN_SENSITIVITY = _env_float("BITSFUSION_SELF_ATTN_SENSITIVITY", 1.25)
FFN_SENSITIVITY = _env_float("BITSFUSION_FFN_SENSITIVITY", 1.00)
CONV_SENSITIVITY = _env_float("BITSFUSION_CONV_SENSITIVITY", 1.10)
TIMESTEP_BUCKET_NAMES = ("early", "mid", "late")
TIMESTEP_BUCKET_WEIGHTS = _parse_float_sequence(os.environ.get("BITSFUSION_TIMESTEP_BUCKET_WEIGHTS", "1.0,1.0,1.0"), min_value=0.0)
if len(TIMESTEP_BUCKET_WEIGHTS) != len(TIMESTEP_BUCKET_NAMES):
    raise ValueError(
        "BITSFUSION_TIMESTEP_BUCKET_WEIGHTS must contain exactly "
        f"{len(TIMESTEP_BUCKET_NAMES)} values: {','.join(TIMESTEP_BUCKET_NAMES)}"
    )
if not any(float(x) > 0 for x in TIMESTEP_BUCKET_WEIGHTS):
    raise ValueError("BITSFUSION_TIMESTEP_BUCKET_WEIGHTS must contain at least one positive value")
CLIP_OPT_OUT_CHUNK = _env_int("BITSFUSION_CLIP_OPT_OUT_CHUNK", 256)
if CLIP_OPT_OUT_CHUNK < 1:
    raise ValueError(f"BITSFUSION_CLIP_OPT_OUT_CHUNK must be >= 1, got {CLIP_OPT_OUT_CHUNK}")
if CALIB_MAX_ROWS_PER_LAYER < 1:
    raise ValueError(f"BITSFUSION_CALIB_MAX_ROWS_PER_LAYER must be >= 1, got {CALIB_MAX_ROWS_PER_LAYER}")
if CALIB_STEPS < 1:
    raise ValueError(f"BITSFUSION_CALIB_STEPS must be >= 1, got {CALIB_STEPS}")
FAST_SMOKE = _env_bool("BITSFUSION_FAST_SMOKE", False)

if FAST_SMOKE:
    EVAL_PROMPT_COUNT = min(EVAL_PROMPT_COUNT, 1)
    EVAL_SEEDS = EVAL_SEEDS[:1]
    CALIB_PROMPT_COUNT = min(CALIB_PROMPT_COUNT, 1)
    CALIB_SEEDS = CALIB_SEEDS[:1]
    CALIB_STEPS = min(CALIB_STEPS, 2)
    CALIB_MAX_ROWS_PER_LAYER = min(CALIB_MAX_ROWS_PER_LAYER, 256)
    TARGET_BPPS = TARGET_BPPS[:1]
    LAYER_BITS_LIST = LAYER_BITS_LIST[: min(2, len(LAYER_BITS_LIST))]

if not TARGET_BPPS:
    raise ValueError("BITSFUSION_TARGET_BPPS must contain at least one target bpp.")
if not LAYER_BITS_LIST:
    raise ValueError("BITSFUSION_LAYER_BITS must contain at least one integer bit-width.")
if 1 in LAYER_BITS_LIST and not BALANCED_INTEGER:
    raise ValueError("BITSFUSION_LAYER_BITS includes 1; set BITSFUSION_BALANCED_INTEGER=1 for the 1-bit proxy quantizer.")
if ACTIVATION_BITS < 0:
    raise ValueError(f"BITSFUSION_ACT_BITS must be >= 0, got {ACTIVATION_BITS}")
if ACTIVATION_BITS == 1:
    raise ValueError("BITSFUSION_ACT_BITS=1 is not supported by the symmetric activation fake-quantizer.")

valid_activation_schemes = {"dynamic_per_tensor", "dynamic_per_sample"}
if ACTIVATION_SCHEME not in valid_activation_schemes:
    raise ValueError(
        f"Unsupported BITSFUSION_ACT_SCHEME={ACTIVATION_SCHEME!r}; "
        f"expected one of {sorted(valid_activation_schemes)}"
    )

valid_allocation_policies = {"greedy", "semantic_sensitivity_greedy"}
if ALLOCATION_POLICY not in valid_allocation_policies:
    raise ValueError(
        f"Unsupported BITSFUSION_ALLOCATION_POLICY={ALLOCATION_POLICY!r}; "
        f"expected one of {sorted(valid_allocation_policies)}"
    )

valid_target_modes = {"hardcoded_midblock", "all_midblock_linear", "all_midblock", "single_largest_linear", "single_named_linear"}
if TARGET_MODE not in valid_target_modes:
    raise ValueError(f"Unsupported BITSFUSION_TARGET_MODE={TARGET_MODE!r}; expected one of {sorted(valid_target_modes)}")


def _pickscore_dataset_roots(env_name: str) -> List[Path]:
    roots: List[Path] = []
    for value in (os.environ.get(env_name), os.environ.get("PICKSCORE_DATA_DIR")):
        if value:
            roots.append(Path(value).expanduser())

    roots.extend(
        [
            Path.cwd(),
            SCRIPT_DIR,
            PROJECT_ROOT,
            PROJECT_ROOT / "vqdm_ref_repo" / "dataset",
            Path.cwd() / "vqdm_ref_repo" / "dataset",
            PROJECT_ROOT / "dataset",
            Path.cwd() / "dataset",
        ]
    )

    unique_roots: List[Path] = []
    seen: set[str] = set()
    for root in roots:
        try:
            key = str(root.resolve()) if root.exists() else str(root)
        except Exception:
            key = str(root)
        if key not in seen:
            seen.add(key)
            unique_roots.append(root)
    return unique_roots


def _resolve_pickscore_csv(filename: str, env_name: str = "BITSFUSION_PICKSCORE_DATA_DIR") -> Path:
    checked: List[Path] = []
    for root in _pickscore_dataset_roots(env_name):
        candidate = root / filename
        checked.append(candidate)
        if candidate.exists():
            return candidate
    checked_text = "\n  - ".join(str(path) for path in checked)
    raise FileNotFoundError(
        f"Required PickScore CSV was not found: {filename}\n"
        f"Set {env_name} or PICKSCORE_DATA_DIR to the directory with PickScore CSV files.\n"
        f"Checked:\n  - {checked_text}"
    )


def load_pickscore_prompts(nsamples: int, *, seed: int = 0, eval_mode: bool = True) -> List[str]:
    filename = "pickscore_filtered_prompts_val.csv" if eval_mode else "pickscore_filtered_prompts_train.csv"
    csv_path = _resolve_pickscore_csv(filename)
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or "caption" not in reader.fieldnames:
            raise ValueError(f"PickScore CSV must contain a 'caption' column: {csv_path}")
        data = [row["caption"].strip() for row in reader if row.get("caption") and row["caption"].strip()]

    if not data:
        raise ValueError(f"PickScore CSV contains no non-empty captions: {csv_path}")
    if nsamples > len(data):
        raise ValueError(f"Requested {nsamples} prompts, but {csv_path} contains only {len(data)} captions")

    rng = np.random.default_rng(seed)
    idx = rng.choice(len(data), size=nsamples, replace=False)
    prompts = [str(data[int(i)]) for i in idx]
    print(f"Using PickScore CSV: {csv_path} ({nsamples}/{len(data)} prompts, seed={seed}, eval_mode={eval_mode})")
    return prompts


def iter_midblock_quant_layers(
    unet: nn.Module,
    *,
    include_linear: bool = True,
    include_conv: bool = True,
    min_features: int = 16,
) -> Dict[str, nn.Module]:
    layers: Dict[str, nn.Module] = {}
    for name, module in unet.mid_block.named_modules():
        if not name:
            continue
        full_name = f"mid_block.{name}"
        if include_linear and isinstance(module, nn.Linear):
            if module.in_features >= min_features and module.out_features >= min_features:
                layers[full_name] = module
        elif include_conv and isinstance(module, nn.Conv2d):
            if module.in_channels >= min_features and module.out_channels >= min_features:
                layers[full_name] = module
    return layers


def select_target_layers(unet: nn.Module) -> Dict[str, nn.Module]:
    all_layers = iter_midblock_quant_layers(unet, include_linear=True, include_conv=True)
    linear_layers = {name: layer for name, layer in all_layers.items() if isinstance(layer, nn.Linear)}
    conv_layers = {name: layer for name, layer in all_layers.items() if isinstance(layer, nn.Conv2d)}

    if TARGET_MODE == "hardcoded_midblock":
        target_names = list(HARDCODED_LINEAR_TARGET_NAMES)
        if INCLUDE_CONV:
            target_names += list(HARDCODED_CONV_TARGET_NAMES)
        missing = [name for name in target_names if name not in all_layers]
        if missing:
            raise RuntimeError("Hardcoded target layers were not found: " + ", ".join(missing))
        return {name: all_layers[name] for name in target_names}

    if TARGET_MODE == "all_midblock_linear":
        if not linear_layers:
            raise RuntimeError("No Linear layers found in mid_block")
        selected = dict(linear_layers)
        if INCLUDE_CONV:
            selected.update(conv_layers)
        return selected

    if TARGET_MODE == "all_midblock":
        if not all_layers:
            raise RuntimeError("No quantizable Linear/Conv2d layers found in mid_block")
        return dict(all_layers)

    if TARGET_MODE == "single_largest_linear":
        if not linear_layers:
            raise RuntimeError("No Linear layers found in mid_block")
        largest_name = max(linear_layers, key=lambda name: linear_layers[name].weight.numel())
        print(f"Selected largest Linear target layer: {largest_name}")
        return {largest_name: linear_layers[largest_name]}

    if TARGET_MODE == "single_named_linear":
        if SINGLE_TARGET_LAYER not in linear_layers:
            largest_name = max(linear_layers, key=lambda name: linear_layers[name].weight.numel())
            print(f"Requested layer not found; selected largest Linear instead: {largest_name}")
            return {largest_name: linear_layers[largest_name]}
        return {SINGLE_TARGET_LAYER: linear_layers[SINGLE_TARGET_LAYER]}

    raise AssertionError(f"Unhandled target mode: {TARGET_MODE}")


def _split_midblock_name(full_name: str) -> List[str]:
    if not full_name.startswith("mid_block."):
        raise ValueError(f"Expected a mid_block-prefixed layer name, got {full_name!r}")
    return full_name[len("mid_block.") :].split(".")


def _get_child(module: nn.Module, token: str) -> nn.Module:
    if token.isdigit() and isinstance(module, (nn.Sequential, nn.ModuleList)):
        return module[int(token)]
    return getattr(module, token)


def _set_child(module: nn.Module, token: str, child: nn.Module):
    if token.isdigit() and isinstance(module, (nn.Sequential, nn.ModuleList)):
        module[int(token)] = child
    else:
        setattr(module, token, child)


def replace_midblock_submodule(mid_block: nn.Module, full_name: str, new_module: nn.Module):
    parts = _split_midblock_name(full_name)
    parent = mid_block
    for token in parts[:-1]:
        parent = _get_child(parent, token)
    _set_child(parent, parts[-1], new_module)


def _layer_policy_group(full_name: str, layer: nn.Module) -> str:
    name = full_name.lower()
    if isinstance(layer, nn.Conv2d):
        if ".resnets." in name:
            return "resnet_conv"
        if ".attentions." in name:
            return "attention_conv"
        return "conv"
    if ".attn2." in name:
        if any(token in name for token in (".to_q", ".to_k", ".to_v")):
            return "cross_attention_qkv"
        return "cross_attention_out"
    if ".attn1." in name:
        if any(token in name for token in (".to_q", ".to_k", ".to_v")):
            return "self_attention_qkv"
        return "self_attention_out"
    if ".ff." in name or "ff.net" in name:
        return "ffn"
    return "linear"


def _layer_sensitivity_multiplier(full_name: str, layer: nn.Module) -> float:
    if ALLOCATION_POLICY == "greedy":
        return 1.0
    group = _layer_policy_group(full_name, layer)
    if group.startswith("cross_attention"):
        return float(CROSS_ATTN_SENSITIVITY)
    if group.startswith("self_attention") or group == "attention_conv":
        return float(SELF_ATTN_SENSITIVITY)
    if group == "ffn":
        return float(FFN_SENSITIVITY)
    if isinstance(layer, nn.Conv2d):
        return float(CONV_SENSITIVITY)
    return 1.0


def _min_bit_for_layer(full_name: str, layer: nn.Module, allowed_bits: Sequence[int]) -> int:
    allowed = sorted(int(x) for x in allowed_bits)
    if not allowed:
        raise ValueError("Expected at least one allowed bit-width")
    if not SEMANTIC_PROTECT or ALLOCATION_POLICY == "greedy":
        return allowed[0]
    group = _layer_policy_group(full_name, layer)
    sensitive = group.startswith("cross_attention") or group.startswith("self_attention")
    if not sensitive:
        return allowed[0]
    viable = [bits for bits in allowed if bits >= int(SEMANTIC_MIN_BITS)]
    return int(viable[0]) if viable else allowed[-1]


def _timestep_bucket_from_step(step_idx: int) -> str:
    denom = max(int(CALIB_STEPS), 1)
    frac = float(int(step_idx) % denom) / float(denom)
    if frac < 1.0 / 3.0:
        return "early"
    if frac < 2.0 / 3.0:
        return "mid"
    return "late"


def _timestep_bucket_weights_dict() -> Dict[str, float]:
    return {name: float(weight) for name, weight in zip(TIMESTEP_BUCKET_NAMES, TIMESTEP_BUCKET_WEIGHTS)}


def _symmetric_quant_bounds(bits: int) -> Tuple[int, int]:
    if int(bits) <= 1:
        return -1, 1
    qmax = (1 << (int(bits) - 1)) - 1
    qmin = -qmax
    return qmin, qmax


def _binary_sign(x: torch.Tensor) -> torch.Tensor:
    return torch.where(x >= 0, torch.ones_like(x), -torch.ones_like(x))


@torch.no_grad()
def fake_quant_symmetric_activation(x: torch.Tensor, bits: int, scheme: str = "dynamic_per_tensor") -> torch.Tensor:
    if int(bits) >= 16:
        return x
    qmin, qmax = _symmetric_quant_bounds(bits)
    x_float = x.detach().float()
    scheme = str(scheme).strip().lower()
    if scheme == "dynamic_per_tensor" or x_float.ndim <= 1:
        max_abs = x_float.abs().amax().clamp_min(1e-12)
    elif scheme == "dynamic_per_sample":
        reduce_dims = tuple(range(1, x_float.ndim))
        max_abs = x_float.abs().amax(dim=reduce_dims, keepdim=True).clamp_min(1e-12)
    else:
        raise ValueError(f"Unsupported activation quantization scheme: {scheme!r}")
    scale = max_abs / float(qmax)
    q = torch.round(x_float / scale).clamp(qmin, qmax)
    return (q * scale).to(dtype=x.dtype, device=x.device)


@torch.no_grad()
def quantize_weight_per_output_channel(
    weight: torch.Tensor,
    bits: int,
    *,
    quant_device: str | torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    q_device = torch.device(quant_device)
    w = weight.detach().to(device=q_device, dtype=torch.float32, non_blocking=(q_device.type == "cuda"))
    if int(bits) >= 16:
        scale_shape = [int(w.shape[0])] + [1] * (w.ndim - 1)
        scales = torch.ones(scale_shape, dtype=torch.float32, device=w.device)
        return w.clone(), scales
    if int(bits) == 1:
        reduce_dims = tuple(range(1, w.ndim))
        scales = w.abs().mean(dim=reduce_dims, keepdim=True).clamp_min(1e-12)
        q = _binary_sign(w)
        return (q * scales).float(), scales.float()

    qmin, qmax = _symmetric_quant_bounds(bits)
    reduce_dims = tuple(range(1, w.ndim))
    max_abs = w.abs().amax(dim=reduce_dims, keepdim=True).clamp_min(1e-12)
    scales = max_abs / float(qmax)
    q = torch.round(w / scales).clamp(qmin, qmax)
    return (q * scales).float(), scales.float()


def _scale_shape_for_weight(weight: torch.Tensor) -> List[int]:
    return [int(weight.shape[0])] + [1] * (weight.ndim - 1)


@torch.no_grad()
def quantize_weight_with_scales(
    weight: torch.Tensor,
    bits: int,
    scales: torch.Tensor,
    *,
    quant_device: str | torch.device,
) -> torch.Tensor:
    q_device = torch.device(quant_device)
    w = weight.detach().to(device=q_device, dtype=torch.float32, non_blocking=(q_device.type == "cuda"))
    if int(bits) >= 16:
        return w.clone()
    qmin, qmax = _symmetric_quant_bounds(bits)
    s = scales.detach().to(device=q_device, dtype=torch.float32, non_blocking=(q_device.type == "cuda"))
    s = s.reshape(_scale_shape_for_weight(w)).clamp_min(1e-12)
    if int(bits) == 1:
        return (_binary_sign(w) * s).float()
    q = torch.round(w / s).clamp(qmin, qmax)
    return (q * s).float()


@torch.no_grad()
def optimize_bitsfusion_weight_clip_ratio_grid(
    weight: torch.Tensor,
    calib_rows: Optional[torch.Tensor],
    bits: int,
    *,
    quant_device: str | torch.device,
    clip_ratio_grid: Sequence[float] = CLIP_RATIO_GRID,
    out_chunk: int = CLIP_OPT_OUT_CHUNK,
    calib_bucket_rows: Optional[Dict[str, torch.Tensor]] = None,
    bucket_weights: Sequence[float] = TIMESTEP_BUCKET_WEIGHTS,
    layer_policy_group: str = "unknown",
    sensitivity_multiplier: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any]]:
    """Choose per-output-channel clipping ratios using calibration inputs.

    Objective for each output channel j:
        mean_x ((x @ (q_clip(w_j) - w_j)) ** 2)

    The stored quantization parameter is still only the per-output-channel
    scale. The selected clipping ratio is folded into that scale, so the bpp is
    the same as ordinary scalar PTQ with per-channel scales.
    """

    q_device = torch.device(quant_device)
    w = weight.detach().to(device=q_device, dtype=torch.float32, non_blocking=(q_device.type == "cuda"))
    base_q_weight, base_scales = quantize_weight_per_output_channel(w, bits, quant_device=q_device)
    out_features = int(w.shape[0])
    flat_in_features = int(w[0].numel())
    timestep_weight_map = {name: float(weight) for name, weight in zip(TIMESTEP_BUCKET_NAMES, bucket_weights)}

    fallback_reason: Optional[str] = None
    if int(bits) >= 16:
        fallback_reason = "bits>=16"
    elif calib_rows is None or int(calib_rows.numel()) == 0:
        fallback_reason = "missing_calibration_rows"
    elif int(calib_rows.shape[-1]) != flat_in_features:
        fallback_reason = f"calibration_dim_mismatch:{int(calib_rows.shape[-1])}!={flat_in_features}"

    if fallback_reason is not None:
        clip_ratio = torch.ones(out_features, dtype=torch.float32, device=q_device)
        stats = {
            "calib_rows": 0 if calib_rows is None else int(calib_rows.shape[0]),
            "timestep_bucket_rows": {},
            "timestep_bucket_weights": timestep_weight_map,
            "out_features": out_features,
            "flat_in_features": flat_in_features,
            "clip_ratio_grid": list(float(x) for x in clip_ratio_grid),
            "layer_policy_group": layer_policy_group,
            "sensitivity_multiplier": float(sensitivity_multiplier),
            "balanced_integer": bool(BALANCED_INTEGER),
            "fallback_reason": fallback_reason,
            "clip_ratio_mean": 1.0,
            "clip_ratio_std": 0.0,
            "clip_ratio_min": 1.0,
            "clip_ratio_max": 1.0,
        }
        return base_q_weight, base_scales, clip_ratio, stats

    bucket_items: List[Tuple[str, torch.Tensor, float]] = []
    if calib_bucket_rows:
        for bucket_name in TIMESTEP_BUCKET_NAMES:
            rows = calib_bucket_rows.get(bucket_name)
            weight_value = float(timestep_weight_map.get(bucket_name, 0.0))
            if rows is None or weight_value <= 0.0 or int(rows.numel()) == 0:
                continue
            if int(rows.shape[-1]) != flat_in_features:
                continue
            x_bucket = rows.detach().to(device=q_device, dtype=torch.float32, non_blocking=(q_device.type == "cuda"))
            x_bucket = x_bucket.reshape(int(x_bucket.shape[0]), flat_in_features).contiguous()
            bucket_items.append((bucket_name, x_bucket, weight_value))

    if not bucket_items:
        x_all = calib_rows.detach().to(device=q_device, dtype=torch.float32, non_blocking=(q_device.type == "cuda"))
        x_all = x_all.reshape(int(x_all.shape[0]), flat_in_features).contiguous()
        bucket_items.append(("all", x_all, 1.0))

    w2 = w.reshape(out_features, flat_in_features).contiguous()
    base_scales2 = base_scales.reshape(out_features, 1).contiguous()

    if int(bits) == 1:
        sign2 = _binary_sign(w2).contiguous()
        best_scales = torch.empty((out_features,), dtype=torch.float32, device=q_device)
        best_err = torch.empty((out_features,), dtype=torch.float32, device=q_device)
        chunk_size = max(1, int(out_chunk))

        for start in range(0, out_features, chunk_size):
            end = min(start + chunk_size, out_features)
            sign_chunk = sign2[start:end]
            w_chunk = w2[start:end]
            fallback_scale = w_chunk.abs().mean(dim=1).clamp_min(1e-12)
            numerator = torch.zeros((end - start,), dtype=torch.float32, device=q_device)
            denominator = torch.zeros((end - start,), dtype=torch.float32, device=q_device)

            for _bucket_name, x_bucket, bucket_weight in bucket_items:
                x_sign = x_bucket @ sign_chunk.t().contiguous()
                x_weight = x_bucket @ w_chunk.t().contiguous()
                weight_value = float(bucket_weight)
                numerator = numerator + weight_value * (x_sign * x_weight).mean(dim=0)
                denominator = denominator + weight_value * x_sign.square().mean(dim=0)

            scale = numerator.clamp_min(0.0) / denominator.clamp_min(1e-12)
            scale = torch.where(denominator > 0.0, scale, fallback_scale).clamp_min(1e-12)
            q_w_chunk = sign_chunk * scale.reshape(-1, 1)
            delta = (q_w_chunk - w_chunk).t().contiguous()

            err = torch.zeros((end - start,), dtype=torch.float32, device=q_device)
            denom = 0.0
            for _bucket_name, x_bucket, bucket_weight in bucket_items:
                err = err + float(bucket_weight) * (x_bucket @ delta).square().mean(dim=0)
                denom += float(bucket_weight)
            best_scales[start:end] = scale
            best_err[start:end] = err / max(denom, 1e-12)

        chosen_scales = best_scales.reshape(_scale_shape_for_weight(w)).contiguous()
        q_weight = (_binary_sign(w) * chosen_scales).float()
        max_abs = w.abs().amax(dim=tuple(range(1, w.ndim)), keepdim=True).clamp_min(1e-12)
        scale_ratio = (chosen_scales / max_abs).reshape(out_features).detach().float().cpu()
        bucket_row_counts = {bucket_name: int(rows.shape[0]) for bucket_name, rows, _weight in bucket_items}
        stats = {
            "calib_rows": int(sum(bucket_row_counts.values())),
            "timestep_bucket_rows": bucket_row_counts,
            "timestep_bucket_weights": timestep_weight_map,
            "out_features": out_features,
            "flat_in_features": flat_in_features,
            "clip_ratio_grid": list(float(v) for v in clip_ratio_grid),
            "layer_policy_group": layer_policy_group,
            "sensitivity_multiplier": float(sensitivity_multiplier),
            "balanced_integer": bool(BALANCED_INTEGER),
            "fallback_reason": None,
            "clip_ratio_mean": float(scale_ratio.mean().item()),
            "clip_ratio_std": float(scale_ratio.std(unbiased=False).item()),
            "clip_ratio_min": float(scale_ratio.min().item()),
            "clip_ratio_max": float(scale_ratio.max().item()),
            "mean_reconstruction_mse": float(best_err.detach().float().mean().cpu().item()),
        }
        return q_weight, chosen_scales, scale_ratio.to(q_device), stats

    qmin, qmax = _symmetric_quant_bounds(bits)
    candidates = torch.tensor(list(float(v) for v in clip_ratio_grid), dtype=torch.float32, device=q_device).clamp_min(1e-12)

    best_clip_ratio = torch.ones(out_features, dtype=torch.float32, device=q_device)
    best_err = torch.full((out_features,), float("inf"), dtype=torch.float32, device=q_device)
    chunk_size = max(1, int(out_chunk))

    for start in range(0, out_features, chunk_size):
        end = min(start + chunk_size, out_features)
        w_chunk = w2[start:end]
        scale_chunk = base_scales2[start:end]
        for ratio_value in candidates:
            scales = scale_chunk * ratio_value
            q = torch.round(w_chunk / scales).clamp(qmin, qmax)
            q_w = q * scales
            delta = (q_w - w_chunk).t().contiguous()
            err = torch.zeros((end - start,), dtype=torch.float32, device=q_device)
            denom = 0.0
            for _bucket_name, x_bucket, bucket_weight in bucket_items:
                err = err + float(bucket_weight) * (x_bucket @ delta).square().mean(dim=0)
                denom += float(bucket_weight)
            err = err / max(denom, 1e-12)
            better = err < best_err[start:end]
            if bool(better.any()):
                best_err[start:end] = torch.where(better, err, best_err[start:end])
                best_clip_ratio[start:end] = torch.where(better, ratio_value.expand_as(err), best_clip_ratio[start:end])

    chosen_scales = (base_scales2.squeeze(1) * best_clip_ratio).reshape(_scale_shape_for_weight(w)).contiguous()
    q_weight = quantize_weight_with_scales(w, bits, chosen_scales, quant_device=q_device)
    ratio_cpu = best_clip_ratio.detach().float().cpu()
    bucket_row_counts = {bucket_name: int(rows.shape[0]) for bucket_name, rows, _weight in bucket_items}
    stats = {
        "calib_rows": int(sum(bucket_row_counts.values())),
        "timestep_bucket_rows": bucket_row_counts,
        "timestep_bucket_weights": timestep_weight_map,
        "out_features": out_features,
        "flat_in_features": flat_in_features,
        "clip_ratio_grid": list(float(v) for v in clip_ratio_grid),
        "layer_policy_group": layer_policy_group,
        "sensitivity_multiplier": float(sensitivity_multiplier),
        "balanced_integer": bool(BALANCED_INTEGER),
        "fallback_reason": None,
        "clip_ratio_mean": float(ratio_cpu.mean().item()),
        "clip_ratio_std": float(ratio_cpu.std(unbiased=False).item()),
        "clip_ratio_min": float(ratio_cpu.min().item()),
        "clip_ratio_max": float(ratio_cpu.max().item()),
        "mean_reconstruction_mse": float(best_err.detach().float().mean().cpu().item()),
    }
    return q_weight, chosen_scales, best_clip_ratio, stats


class BitsFusionLiteQuantLinear(nn.Module):
    def __init__(
        self,
        source: nn.Linear,
        cfg: BitsFusionLiteConfig,
        *,
        q_weight: torch.Tensor,
        scales: torch.Tensor,
        clip_ratio: torch.Tensor,
        quant_device: str | torch.device,
    ):
        super().__init__()
        q_device = torch.device(quant_device)
        self.register_buffer("weight", q_weight.detach().to(device=q_device, dtype=torch.float32).contiguous())
        self.register_buffer("weight_scales", scales.detach().to(device=q_device, dtype=torch.float32).reshape(-1).contiguous())
        self.register_buffer("clip_ratio", clip_ratio.detach().to(device=q_device, dtype=torch.float32).reshape(-1).contiguous())
        if source.bias is None:
            self.register_buffer("bias", None)
        else:
            self.register_buffer(
                "bias",
                source.bias.detach().to(device=q_device, dtype=torch.float32, non_blocking=(q_device.type == "cuda")).clone(),
            )
        self.in_features = int(source.in_features)
        self.out_features = int(source.out_features)
        self.weight_bits = int(cfg.weight_bits)
        self.activation_bits = int(cfg.activation_bits)
        self.mode = str(cfg.mode)
        self.activation_scheme = str(cfg.activation_scheme)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.activation_bits > 0:
            x = fake_quant_symmetric_activation(x, self.activation_bits, self.activation_scheme)
        bias = None if self.bias is None else self.bias.to(dtype=x.dtype)
        return F.linear(x, self.weight.to(dtype=x.dtype), bias)


class BitsFusionLiteQuantConv2d(nn.Module):
    def __init__(
        self,
        source: nn.Conv2d,
        cfg: BitsFusionLiteConfig,
        *,
        q_weight: torch.Tensor,
        scales: torch.Tensor,
        clip_ratio: torch.Tensor,
        quant_device: str | torch.device,
    ):
        super().__init__()
        q_device = torch.device(quant_device)
        self.register_buffer("weight", q_weight.detach().to(device=q_device, dtype=torch.float32).contiguous())
        self.register_buffer("weight_scales", scales.detach().to(device=q_device, dtype=torch.float32).reshape(-1).contiguous())
        self.register_buffer("clip_ratio", clip_ratio.detach().to(device=q_device, dtype=torch.float32).reshape(-1).contiguous())
        if source.bias is None:
            self.register_buffer("bias", None)
        else:
            self.register_buffer(
                "bias",
                source.bias.detach().to(device=q_device, dtype=torch.float32, non_blocking=(q_device.type == "cuda")).clone(),
            )
        self.in_channels = int(source.in_channels)
        self.out_channels = int(source.out_channels)
        self.kernel_size = tuple(source.kernel_size)
        self.stride = tuple(source.stride)
        self.padding = source.padding if isinstance(source.padding, str) else tuple(source.padding)
        self.dilation = tuple(source.dilation)
        self.groups = int(source.groups)
        self.padding_mode = str(source.padding_mode)
        self.weight_bits = int(cfg.weight_bits)
        self.activation_bits = int(cfg.activation_bits)
        self.mode = str(cfg.mode)
        self.activation_scheme = str(cfg.activation_scheme)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.activation_bits > 0:
            x = fake_quant_symmetric_activation(x, self.activation_bits, self.activation_scheme)
        bias = None if self.bias is None else self.bias.to(dtype=x.dtype)
        weight = self.weight.to(dtype=x.dtype)
        if isinstance(self.padding, str):
            padding = self.padding
        elif self.padding_mode != "zeros":
            x = F.pad(x, self._reversed_padding_repeated_twice, mode=self.padding_mode)
            padding = (0, 0)
        else:
            padding = self.padding
        return F.conv2d(x, weight, bias, self.stride, padding, self.dilation, self.groups)

    @property
    def _reversed_padding_repeated_twice(self):
        return tuple(x for pair in zip(reversed(self.padding), reversed(self.padding)) for x in pair)


def _layer_scale_count(layer: nn.Module) -> int:
    return int(layer.weight.shape[0])


def _layer_quant_bit_cost(layer: nn.Module, bits: int, scale_bits: int) -> int:
    return int(layer.weight.numel()) * int(bits) + _layer_scale_count(layer) * int(scale_bits)


def _candidate_objective(source_weight: torch.Tensor, q_weight: torch.Tensor, opt_stats: Dict[str, Any]) -> float:
    recon = opt_stats.get("mean_reconstruction_mse")
    out_features = max(int(opt_stats.get("out_features", int(source_weight.shape[0]))), 1)
    if recon is not None and math.isfinite(float(recon)):
        return float(recon) * float(out_features)
    diff = q_weight.detach().float().cpu() - source_weight.detach().float().cpu()
    return float(diff.square().mean().item()) * float(source_weight.numel())


@torch.no_grad()
def build_bitsfusion_candidate_table(
    target_layers: Dict[str, nn.Module],
    cfg: BitsFusionLiteConfig,
    *,
    calibration_rows: Dict[str, torch.Tensor],
    calibration_bucket_rows: Optional[Dict[str, Dict[str, torch.Tensor]]] = None,
    quant_device: str | torch.device,
) -> Dict[str, Dict[int, Dict[str, Any]]]:
    q_device = torch.device(quant_device)
    candidates: Dict[str, Dict[int, Dict[str, Any]]] = {}
    for full_name, source_layer in target_layers.items():
        rows = calibration_rows.get(full_name)
        layer_group = _layer_policy_group(full_name, source_layer)
        sensitivity_multiplier = _layer_sensitivity_multiplier(full_name, source_layer)
        layer_candidates: Dict[int, Dict[str, Any]] = {}
        for bits in cfg.allowed_bits:
            q_weight, scales, clip_ratio, opt_stats = optimize_bitsfusion_weight_clip_ratio_grid(
                source_layer.weight,
                rows,
                int(bits),
                quant_device=q_device,
                clip_ratio_grid=CLIP_RATIO_GRID,
                out_chunk=CLIP_OPT_OUT_CHUNK,
                calib_bucket_rows=(calibration_bucket_rows or {}).get(full_name, {}),
                bucket_weights=TIMESTEP_BUCKET_WEIGHTS,
                layer_policy_group=layer_group,
                sensitivity_multiplier=sensitivity_multiplier,
            )
            opt_stats = {"name": full_name, "layer_type": type(source_layer).__name__, "weight_bits": int(bits), **opt_stats}
            raw_objective = _candidate_objective(source_layer.weight, q_weight, opt_stats)
            objective = float(raw_objective) * float(sensitivity_multiplier)
            layer_candidates[int(bits)] = {
                "q_weight": q_weight,
                "scales": scales,
                "clip_ratio": clip_ratio,
                "opt_stats": opt_stats,
                "raw_objective": float(raw_objective),
                "objective": float(objective),
                "layer_policy_group": layer_group,
                "sensitivity_multiplier": float(sensitivity_multiplier),
                "weight_params": int(source_layer.weight.numel()),
                "scale_count": _layer_scale_count(source_layer),
                "quant_weight_bits": _layer_quant_bit_cost(source_layer, int(bits), cfg.weight_scale_bits),
            }
            if opt_stats.get("fallback_reason"):
                log_event(f"BITSFUSION clipping search fallback | {full_name} @ {bits}b: {opt_stats['fallback_reason']}")
        candidates[full_name] = layer_candidates
    return candidates


def allocate_bitsfusion_bits(
    target_layers: Dict[str, nn.Module],
    cfg: BitsFusionLiteConfig,
    candidates: Dict[str, Dict[int, Dict[str, Any]]],
) -> Tuple[Dict[str, int], Dict[str, Any]]:
    allowed_bits = sorted(int(x) for x in cfg.allowed_bits)
    if not allowed_bits:
        raise ValueError("BitsFusion allocation requires at least one allowed bit-width.")

    total_weight_params = sum(int(layer.weight.numel()) for layer in target_layers.values())
    target_budget_bits = float(cfg.target_bpp) * float(max(total_weight_params, 1))
    min_bits_by_layer = {
        name: _min_bit_for_layer(name, layer, allowed_bits)
        for name, layer in target_layers.items()
    }
    choice_index = {
        name: allowed_bits.index(int(min_bits_by_layer[name]))
        for name in target_layers
    }

    def chosen_bits_for(name: str) -> int:
        return allowed_bits[choice_index[name]]

    def total_quant_bits() -> int:
        return sum(
            int(candidates[name][chosen_bits_for(name)]["quant_weight_bits"])
            for name in target_layers
        )

    current_bits = total_quant_bits()
    allocation_history: List[Dict[str, Any]] = []
    status = "ok"
    if float(current_bits) > target_budget_bits + 1e-6:
        status = "min_bits_exceed_target"
        return {name: chosen_bits_for(name) for name in target_layers}, {
            "status": status,
            "target_budget_bits": target_budget_bits,
            "used_quant_bits": current_bits,
            "target_bpp": float(cfg.target_bpp),
            "used_bpp": current_bits / max(total_weight_params, 1),
            "allocation_policy": ALLOCATION_POLICY,
            "semantic_protect": bool(SEMANTIC_PROTECT),
            "semantic_min_bits": int(SEMANTIC_MIN_BITS),
            "min_bits_by_layer": {name: int(bits) for name, bits in min_bits_by_layer.items()},
            "history": allocation_history,
        }

    while True:
        best_upgrade: Optional[Dict[str, Any]] = None
        for name in target_layers:
            idx = choice_index[name]
            if idx + 1 >= len(allowed_bits):
                continue
            cur_bits = allowed_bits[idx]
            cur_candidate = candidates[name][cur_bits]
            for next_idx in range(idx + 1, len(allowed_bits)):
                next_bits = allowed_bits[next_idx]
                next_candidate = candidates[name][next_bits]
                delta_bits = int(next_candidate["quant_weight_bits"]) - int(cur_candidate["quant_weight_bits"])
                if delta_bits <= 0:
                    continue
                if float(current_bits + delta_bits) > target_budget_bits + 1e-6:
                    continue
                benefit = float(cur_candidate["objective"]) - float(next_candidate["objective"])
                if benefit <= 0.0:
                    continue
                density = benefit / float(delta_bits)
                proposal = {
                    "name": name,
                    "from_bits": int(cur_bits),
                    "to_bits": int(next_bits),
                    "from_index": int(idx),
                    "to_index": int(next_idx),
                    "delta_bits": int(delta_bits),
                    "benefit": float(benefit),
                    "benefit_per_bit": float(density),
                }
                if best_upgrade is None:
                    best_upgrade = proposal
                else:
                    key = (proposal["benefit_per_bit"], proposal["benefit"], -proposal["delta_bits"], proposal["name"])
                    old = (
                        best_upgrade["benefit_per_bit"],
                        best_upgrade["benefit"],
                        -best_upgrade["delta_bits"],
                        best_upgrade["name"],
                    )
                    if key > old:
                        best_upgrade = proposal

        if best_upgrade is None:
            break

        choice_index[best_upgrade["name"]] = int(best_upgrade["to_index"])
        current_bits += int(best_upgrade["delta_bits"])
        best_upgrade.pop("from_index", None)
        best_upgrade.pop("to_index", None)
        best_upgrade["used_quant_bits"] = int(current_bits)
        best_upgrade["used_bpp"] = float(current_bits / max(total_weight_params, 1))
        allocation_history.append(best_upgrade)

    return {name: chosen_bits_for(name) for name in target_layers}, {
        "status": status,
        "target_budget_bits": target_budget_bits,
        "used_quant_bits": int(current_bits),
        "target_bpp": float(cfg.target_bpp),
        "used_bpp": float(current_bits / max(total_weight_params, 1)),
        "allocation_policy": ALLOCATION_POLICY,
        "semantic_protect": bool(SEMANTIC_PROTECT),
        "semantic_min_bits": int(SEMANTIC_MIN_BITS),
        "min_bits_by_layer": {name: int(bits) for name, bits in min_bits_by_layer.items()},
        "history": allocation_history,
    }


def bitsfusion_mixed_compression_stats(
    target_layers: Dict[str, nn.Module],
    cfg: BitsFusionLiteConfig,
    candidates: Dict[str, Dict[int, Dict[str, Any]]],
    allocation: Dict[str, int],
    allocation_info: Dict[str, Any],
    *,
    fp_bits: int = 16,
) -> Dict[str, Any]:
    total_weight_params = 0
    total_bias_params = 0
    total_raw_weight_bits = 0
    total_quant_weight_bits = 0
    total_scale_bits = 0
    weighted_raw_bits = 0
    bit_histogram: Dict[str, int] = {}
    per_layer: List[Dict[str, Any]] = []

    for name, layer in target_layers.items():
        bits = int(allocation[name])
        candidate = candidates[name][bits]
        opt_stats = candidate["opt_stats"]
        weight_params = int(layer.weight.numel())
        bias_params = int(layer.bias.numel()) if getattr(layer, "bias", None) is not None else 0
        scale_count = _layer_scale_count(layer)
        scale_bits = scale_count * int(cfg.weight_scale_bits)
        raw_weight_bits = weight_params * bits
        quant_weight_bits = raw_weight_bits + scale_bits

        total_weight_params += weight_params
        total_bias_params += bias_params
        total_raw_weight_bits += raw_weight_bits
        total_quant_weight_bits += quant_weight_bits
        total_scale_bits += scale_bits
        weighted_raw_bits += raw_weight_bits
        bit_histogram[str(bits)] = bit_histogram.get(str(bits), 0) + 1

        per_layer.append(
            {
                "name": name,
                "layer_type": type(layer).__name__,
                "weight_shape": list(layer.weight.shape),
                "weight_params": weight_params,
                "bias_params": bias_params,
                "weight_bits": bits,
                "activation_bits": int(cfg.activation_bits),
                "scale_count": scale_count,
                "raw_weight_bits": raw_weight_bits,
                "scale_bits": scale_bits,
                "quant_weight_bits": quant_weight_bits,
                "weight_bpp_with_scales": quant_weight_bits / max(weight_params, 1),
                "layer_policy_group": str(candidate.get("layer_policy_group", "")),
                "sensitivity_multiplier": float(candidate.get("sensitivity_multiplier", 1.0)),
                "objective": float(candidate["objective"]),
                "raw_objective": float(candidate.get("raw_objective", candidate["objective"])),
                "clip_ratio_mean": float(opt_stats.get("clip_ratio_mean", float("nan"))),
                "clip_ratio_min": float(opt_stats.get("clip_ratio_min", float("nan"))),
                "clip_ratio_max": float(opt_stats.get("clip_ratio_max", float("nan"))),
                "candidate_objectives": {str(b): float(candidates[name][int(b)]["objective"]) for b in cfg.allowed_bits},
            }
        )

    fp_weight_bits = total_weight_params * int(fp_bits)
    return {
        "config": {
            "mode": cfg.mode,
            "target_bpp": float(cfg.target_bpp),
            "allowed_bits": list(int(x) for x in cfg.allowed_bits),
            "activation_bits": int(cfg.activation_bits),
            "weight_scale_bits": int(cfg.weight_scale_bits),
            "activation_scheme": cfg.activation_scheme,
            "allocation_policy": ALLOCATION_POLICY,
            "balanced_integer": bool(BALANCED_INTEGER),
            "semantic_protect": bool(SEMANTIC_PROTECT),
            "semantic_min_bits": int(SEMANTIC_MIN_BITS),
        },
        "allocation": {
            **allocation_info,
            "bit_histogram": bit_histogram,
            "bit_allocation": {name: int(bits) for name, bits in allocation.items()},
            "effective_avg_weight_bits_without_scales": weighted_raw_bits / max(total_weight_params, 1),
            "budget_margin_bits": float(allocation_info.get("target_budget_bits", 0.0)) - float(total_quant_weight_bits),
        },
        "totals": {
            "target_layers": len(target_layers),
            "target_weight_params": total_weight_params,
            "target_bias_params": total_bias_params,
            "target_weight_fp_bits": fp_weight_bits,
            "raw_weight_bits": total_raw_weight_bits,
            "quant_weight_bits": total_quant_weight_bits,
            "scale_bits": total_scale_bits,
            "quant_weight_bpp": total_quant_weight_bits / max(total_weight_params, 1),
            "quant_weight_compression": fp_weight_bits / max(total_quant_weight_bits, 1),
        },
        "per_layer": per_layer,
    }


@torch.no_grad()
def _sample_rows_deterministic(rows: torch.Tensor, max_rows: int) -> torch.Tensor:
    max_rows = int(max_rows)
    if max_rows <= 0 or int(rows.shape[0]) <= max_rows:
        return rows
    idx = torch.linspace(0, int(rows.shape[0]) - 1, steps=max_rows, device=rows.device).round().long()
    return rows.index_select(0, idx)


def _conv_reversed_padding(layer: nn.Conv2d) -> Tuple[int, ...]:
    padding = layer.padding if not isinstance(layer.padding, str) else (0, 0)
    return tuple(x for pair in zip(reversed(padding), reversed(padding)) for x in pair)


@torch.no_grad()
def _activation_input_rows(layer: nn.Module, x: torch.Tensor) -> Optional[torch.Tensor]:
    x = x.detach()
    if isinstance(layer, nn.Linear):
        if x.ndim == 0 or int(x.shape[-1]) != int(layer.in_features):
            return None
        return x.reshape(-1, int(layer.in_features))

    if isinstance(layer, nn.Conv2d):
        if int(layer.groups) != 1:
            return None
        x_float = x.float()
        if layer.padding_mode != "zeros":
            x_float = F.pad(x_float, _conv_reversed_padding(layer), mode=layer.padding_mode)
            padding = (0, 0)
        elif isinstance(layer.padding, str):
            return None
        else:
            padding = tuple(layer.padding)
        unfolded = F.unfold(
            x_float,
            kernel_size=tuple(layer.kernel_size),
            dilation=tuple(layer.dilation),
            padding=padding,
            stride=tuple(layer.stride),
        )
        return unfolded.transpose(1, 2).reshape(-1, int(unfolded.shape[1]))

    return None


@torch.no_grad()
def _run_bitsfusion_calibration_chunk(
    pipe: StableDiffusionPipeline,
    prompts_batch: Sequence[str],
    seeds_batch: Sequence[int],
    *,
    gen_device: str,
):
    prompts_batch = list(prompts_batch)
    seeds_batch = [int(seed) for seed in seeds_batch]
    try:
        generators = [torch.Generator(device=gen_device).manual_seed(seed) for seed in seeds_batch]
        _ = pipe(
            prompts_batch,
            generator=generators,
            num_inference_steps=CALIB_STEPS,
            guidance_scale=GUIDANCE_SCALE,
        ).images
    except RuntimeError as exc:
        if _is_cuda_oom(exc) and len(prompts_batch) > 1:
            _clear_cuda()
            mid = len(prompts_batch) // 2
            log_event(
                f"CUDA OOM in BITSFUSION calibration; splitting batch "
                f"{len(prompts_batch)} -> {mid}+{len(prompts_batch)-mid}"
            )
            _run_bitsfusion_calibration_chunk(pipe, prompts_batch[:mid], seeds_batch[:mid], gen_device=gen_device)
            _run_bitsfusion_calibration_chunk(pipe, prompts_batch[mid:], seeds_batch[mid:], gen_device=gen_device)
            return
        raise


@torch.no_grad()
def collect_bitsfusion_calibration_rows(
    pipe: StableDiffusionPipeline,
    target_layers: Dict[str, nn.Module],
    prompts: Sequence[str],
    seeds: Sequence[int],
) -> Tuple[Dict[str, torch.Tensor], Dict[str, Dict[str, torch.Tensor]], Dict[str, Dict[str, int]]]:
    storage: Dict[str, torch.Tensor] = {}
    bucket_storage: Dict[str, Dict[str, torch.Tensor]] = {}
    bucket_counts: Dict[str, Dict[str, int]] = {}
    skipped: Dict[str, str] = {}
    step_state = {"call_idx": 0, "bucket": "early"}
    max_rows_per_bucket = max(1, int(math.ceil(float(CALIB_MAX_ROWS_PER_LAYER) / float(len(TIMESTEP_BUCKET_NAMES)))))

    def append_rows(name: str, rows: torch.Tensor, bucket_name: str):
        rows = _sample_rows_deterministic(rows.detach(), CALIB_MAX_ROWS_PER_LAYER)
        rows_cpu = rows.to(device="cpu", dtype=torch.float32)
        old = storage.get(name)
        if old is None:
            merged = rows_cpu
        else:
            merged = torch.cat([old, rows_cpu], dim=0)
        storage[name] = _sample_rows_deterministic(merged, CALIB_MAX_ROWS_PER_LAYER).contiguous()

        if bucket_name not in TIMESTEP_BUCKET_NAMES:
            bucket_name = "mid"
        bucket_counts.setdefault(name, {})
        bucket_counts[name][bucket_name] = int(bucket_counts[name].get(bucket_name, 0)) + int(rows_cpu.shape[0])

        layer_buckets = bucket_storage.setdefault(name, {})
        old_bucket = layer_buckets.get(bucket_name)
        if old_bucket is None:
            merged_bucket = rows_cpu
        else:
            merged_bucket = torch.cat([old_bucket, rows_cpu], dim=0)
        layer_buckets[bucket_name] = _sample_rows_deterministic(merged_bucket, max_rows_per_bucket).contiguous()

    hooks = []
    def unet_step_hook(_module, inputs):
        step_idx = int(step_state["call_idx"]) % max(int(CALIB_STEPS), 1)
        step_state["bucket"] = _timestep_bucket_from_step(step_idx)
        step_state["call_idx"] = int(step_state["call_idx"]) + 1

    hooks.append(pipe.unet.register_forward_pre_hook(unet_step_hook))
    for name, layer in target_layers.items():
        def make_hook(layer_name: str, source_layer: nn.Module):
            def hook(_module, inputs, _output):
                if not inputs:
                    return
                try:
                    rows = _activation_input_rows(source_layer, inputs[0])
                except Exception as exc:
                    skipped.setdefault(layer_name, f"{type(exc).__name__}: {exc}")
                    return
                if rows is None or int(rows.numel()) == 0:
                    skipped.setdefault(layer_name, "unsupported_or_empty_input")
                    return
                append_rows(layer_name, rows, str(step_state.get("bucket", "mid")))

            return hook

        hooks.append(layer.register_forward_hook(make_hook(name, layer)))

    pairs = eval_pairs(prompts, seeds)
    if CALIB_GEN_BATCH_ENV > 0:
        calib_batch = min(max(1, int(CALIB_GEN_BATCH_ENV)), max(1, len(pairs)))
        log_event(f"Resolved calibration batch | env batch={calib_batch}, total_items={len(pairs)}")
    else:
        calib_batch = resolve_generation_batch_size(len(pairs), label="BITSFUSION calibration")

    try:
        force_pipeline_device_dtype(pipe, RUNTIME_DEVICE, RUNTIME_DTYPE)
        pipe.set_progress_bar_config(disable=True)
        gen_device = RUNTIME_DEVICE if str(RUNTIME_DEVICE).startswith("cuda") else "cpu"
        for b0 in tqdm(range(0, len(pairs), calib_batch), desc="BITSFUSION calibration", unit="batch"):
            chunk = pairs[b0 : b0 + calib_batch]
            chunk_prompts = [x[0] for x in chunk]
            chunk_seeds = [int(x[1]) for x in chunk]
            t0 = time.perf_counter()
            _run_bitsfusion_calibration_chunk(pipe, chunk_prompts, chunk_seeds, gen_device=gen_device)
            synchronize_cuda()
            log_event(f"BITSFUSION calibration batch | n={len(chunk)}, steps={CALIB_STEPS}, time={time.perf_counter() - t0:.2f}s")
            _clear_cuda()
    finally:
        for handle in hooks:
            handle.remove()

    for name, reason in skipped.items():
        if name not in storage:
            log_event(f"BITSFUSION calibration skipped rows for {name}: {reason}")

    for name, rows in storage.items():
        log_event(f"BITSFUSION calibration rows | {name}: rows={int(rows.shape[0])}, dim={int(rows.shape[1])}")
        buckets = {
            bucket_name: int(bucket_rows.shape[0])
            for bucket_name, bucket_rows in bucket_storage.get(name, {}).items()
        }
        if buckets:
            log_event(f"BITSFUSION timestep buckets | {name}: {buckets}")
    return storage, bucket_storage, bucket_counts


@torch.no_grad()
def build_bitsfusion_lite_midblock(
    pipe_fp: StableDiffusionPipeline,
    target_layers: Dict[str, nn.Module],
    cfg: BitsFusionLiteConfig,
    *,
    calibration_rows: Dict[str, torch.Tensor],
    calibration_bucket_rows: Optional[Dict[str, Dict[str, torch.Tensor]]] = None,
    candidate_table: Optional[Dict[str, Dict[int, Dict[str, Any]]]] = None,
    quant_device: str | torch.device = QUANT_DEVICE,
) -> Tuple[nn.Module, Dict[str, Any]]:
    q_device = torch.device(quant_device)
    mid_block_q = copy.deepcopy(pipe_fp.unet.mid_block).to(q_device, dtype=torch.float32)
    if candidate_table is None:
        candidates = build_bitsfusion_candidate_table(
            target_layers,
            cfg,
            calibration_rows=calibration_rows,
            calibration_bucket_rows=calibration_bucket_rows,
            quant_device=q_device,
        )
    else:
        candidates = candidate_table
    allocation, allocation_info = allocate_bitsfusion_bits(target_layers, cfg, candidates)
    stats = bitsfusion_mixed_compression_stats(target_layers, cfg, candidates, allocation, allocation_info)
    layer_calibration_stats: List[Dict[str, Any]] = []

    for full_name, source_layer in target_layers.items():
        chosen_bits = int(allocation[full_name])
        chosen = candidates[full_name][chosen_bits]
        q_weight = chosen["q_weight"]
        scales = chosen["scales"]
        clip_ratio = chosen["clip_ratio"]
        opt_stats = dict(chosen["opt_stats"])
        opt_stats["chosen"] = True
        opt_stats["objective"] = float(chosen["objective"])
        layer_calibration_stats.append(opt_stats)
        layer_cfg = cfg.with_layer_bits(chosen_bits)

        if isinstance(source_layer, nn.Linear):
            q_layer = BitsFusionLiteQuantLinear(
                source_layer,
                layer_cfg,
                q_weight=q_weight,
                scales=scales,
                clip_ratio=clip_ratio,
                quant_device=q_device,
            ).to(q_device, dtype=torch.float32)
        elif isinstance(source_layer, nn.Conv2d):
            q_layer = BitsFusionLiteQuantConv2d(
                source_layer,
                layer_cfg,
                q_weight=q_weight,
                scales=scales,
                clip_ratio=clip_ratio,
                quant_device=q_device,
            ).to(q_device, dtype=torch.float32)
        else:
            raise TypeError(f"Unsupported target layer type for {full_name}: {type(source_layer).__name__}")
        replace_midblock_submodule(mid_block_q, full_name, q_layer)

    stats["bitsfusion_calibration"] = {
        "calib_prompt_count": int(CALIB_PROMPT_COUNT),
        "calib_seeds": list(int(x) for x in CALIB_SEEDS),
        "calib_steps": int(CALIB_STEPS),
        "allocation_policy": ALLOCATION_POLICY,
        "balanced_integer": bool(BALANCED_INTEGER),
        "semantic_protect": bool(SEMANTIC_PROTECT),
        "semantic_min_bits": int(SEMANTIC_MIN_BITS),
        "timestep_bucket_names": list(TIMESTEP_BUCKET_NAMES),
        "timestep_bucket_weights": _timestep_bucket_weights_dict(),
        "max_rows_per_layer": int(CALIB_MAX_ROWS_PER_LAYER),
        "clip_ratio_grid": list(float(x) for x in CLIP_RATIO_GRID),
        "clip_opt_out_chunk": int(CLIP_OPT_OUT_CHUNK),
        "per_layer": layer_calibration_stats,
    }
    stats["bitsfusion_candidates"] = {
        name: {
            str(bits): {
                "objective": float(candidate["objective"]),
                "raw_objective": float(candidate.get("raw_objective", candidate["objective"])),
                "quant_weight_bits": int(candidate["quant_weight_bits"]),
                "layer_policy_group": str(candidate.get("layer_policy_group", "")),
                "sensitivity_multiplier": float(candidate.get("sensitivity_multiplier", 1.0)),
                "clip_ratio_mean": float(candidate["opt_stats"].get("clip_ratio_mean", float("nan"))),
                "fallback_reason": candidate["opt_stats"].get("fallback_reason"),
            }
            for bits, candidate in sorted(layer_candidates.items())
        }
        for name, layer_candidates in candidates.items()
    }
    return mid_block_q, stats


@torch.no_grad()
def prepare_images_for_clip(imgs, clip_preprocess, clip_model):
    return torch.stack([clip_preprocess(img) for img in imgs], dim=0).to(_module_device(clip_model))


@torch.no_grad()
def prepare_images_for_lpips(imgs, lpips_transform, lpips_model):
    return torch.stack([lpips_transform(img) for img in imgs], dim=0).to(_module_device(lpips_model))


@torch.no_grad()
def compute_pixel_metrics_batch(img_fp_pils, img_test_pils) -> List[Dict[str, float]]:
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
    c1 = 0.01**2
    c2 = 0.03**2
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
def compute_metrics_batch_against_fp(
    img_fp_pils,
    img_test_pils,
    prompts_batch: Sequence[str],
    *,
    clip_model,
    clip_tokenizer,
    clip_preprocess,
    lpips_model,
    lpips_transform,
) -> List[Dict[str, float]]:
    dev_clip = _module_device(clip_model)
    prompts_batch = list(prompts_batch)

    img_fp_clip = prepare_images_for_clip(img_fp_pils, clip_preprocess, clip_model)
    img_test_clip = prepare_images_for_clip(img_test_pils, clip_preprocess, clip_model)
    txt = clip_tokenizer(prompts_batch).to(dev_clip)

    txt_emb = clip_model.encode_text(txt)
    fp_emb = clip_model.encode_image(img_fp_clip.to(dev_clip))
    test_emb = clip_model.encode_image(img_test_clip.to(dev_clip))

    txt_emb = txt_emb / txt_emb.norm(dim=-1, keepdim=True)
    fp_emb = fp_emb / fp_emb.norm(dim=-1, keepdim=True)
    test_emb = test_emb / test_emb.norm(dim=-1, keepdim=True)

    clip_text_fp = (txt_emb * fp_emb).sum(dim=-1).detach().float().cpu().tolist()
    clip_text_test = (txt_emb * test_emb).sum(dim=-1).detach().float().cpu().tolist()
    clip_img_sim = (fp_emb * test_emb).sum(dim=-1).detach().float().cpu().tolist()

    img_fp_lp = prepare_images_for_lpips(img_fp_pils, lpips_transform, lpips_model) * 2 - 1
    img_test_lp = prepare_images_for_lpips(img_test_pils, lpips_transform, lpips_model) * 2 - 1
    lp = lpips_model(img_fp_lp, img_test_lp).view(-1).detach().float().cpu().tolist()
    pix = compute_pixel_metrics_batch(img_fp_pils, img_test_pils)

    return [
        {
            "clip_ti_fp": float(clip_text_fp[i]),
            "clip_ti": float(clip_text_test[i]),
            "delta_clip_ti": float(clip_text_test[i] - clip_text_fp[i]),
            "clip_ii": float(clip_img_sim[i]),
            "lpips": float(lp[i]),
            **pix[i],
        }
        for i in range(len(prompts_batch))
    ]


def eval_pairs(prompts: Sequence[str], seeds: Sequence[int]) -> List[Tuple[str, int]]:
    return [(str(prompt), int(seed)) for prompt in prompts for seed in seeds]


def fp_cache_key(prompt: str, seed: int) -> Tuple[str, int, int, float]:
    return (str(prompt), int(seed), int(NUM_INFERENCE_STEPS), float(GUIDANCE_SCALE))


def resolve_generation_batch_size(total_items: Optional[int] = None, label: str = "generation") -> int:
    if GEN_BATCH_ENV > 0:
        batch = int(GEN_BATCH_ENV)
        reason = "env"
    elif not torch.cuda.is_available():
        batch = 1
        reason = "cpu"
    else:
        info = _cuda_mem_info(RUNTIME_DEVICE) or {"free_gib": 0.0, "total_gib": 0.0}
        free_gib = float(info["free_gib"])
        total_gib = float(info["total_gib"])
        usable_gib = max(0.0, free_gib - float(GPU_RESERVE_GIB)) * float(GPU_TARGET_MEM_FRACTION)
        batch = int(max(1.0, usable_gib / max(float(GEN_ESTIMATED_GIB_PER_IMAGE), 1e-6)))
        batch = max(int(GEN_BATCH_MIN), batch)
        cap = GEN_BATCH_MAX if GEN_BATCH_MAX > 0 else (64 if total_gib >= 70.0 else 32)
        batch = min(batch, cap)
        reason = (
            f"auto free={free_gib:.1f}GiB total={total_gib:.1f}GiB "
            f"usable={usable_gib:.1f}GiB est_per_img={GEN_ESTIMATED_GIB_PER_IMAGE:.2f}GiB "
            f"target_fraction={GPU_TARGET_MEM_FRACTION:.2f} reserve={GPU_RESERVE_GIB:.1f}GiB cap={cap}"
        )
    if total_items is not None and int(total_items) > 0:
        batch = min(batch, int(total_items))
    batch = max(1, int(batch))
    log_event(f"Resolved generation batch | {label}: batch={batch}, total_items={total_items}, reason={reason}")
    return batch


@torch.no_grad()
def generate_image(
    pipe,
    prompt: str,
    seed: int,
    *,
    device: str = RUNTIME_DEVICE,
    dtype: torch.dtype = RUNTIME_DTYPE,
):
    gen_device = device if str(device).startswith("cuda") else "cpu"
    generator = torch.Generator(device=gen_device).manual_seed(int(seed))
    force_pipeline_device_dtype(pipe, device, dtype)
    pipe.set_progress_bar_config(disable=True)
    return pipe(
        prompt,
        generator=generator,
        num_inference_steps=NUM_INFERENCE_STEPS,
        guidance_scale=GUIDANCE_SCALE,
    ).images[0]


@torch.no_grad()
def generate_images_batched(
    pipe,
    prompts_batch: Sequence[str],
    seeds_batch: Sequence[int],
    *,
    device: str = RUNTIME_DEVICE,
    dtype: torch.dtype = RUNTIME_DTYPE,
):
    prompts_batch = list(prompts_batch)
    seeds_batch = [int(seed) for seed in seeds_batch]
    if len(prompts_batch) != len(seeds_batch):
        raise ValueError("prompts_batch and seeds_batch must have the same length")
    if not prompts_batch:
        return []
    if len(prompts_batch) == 1:
        return [generate_image(pipe, prompts_batch[0], seeds_batch[0], device=device, dtype=dtype)]

    gen_device = device if str(device).startswith("cuda") else "cpu"
    try:
        generators = [torch.Generator(device=gen_device).manual_seed(seed) for seed in seeds_batch]
        force_pipeline_device_dtype(pipe, device, dtype)
        pipe.set_progress_bar_config(disable=True)
        images = pipe(
            prompts_batch,
            generator=generators,
            num_inference_steps=NUM_INFERENCE_STEPS,
            guidance_scale=GUIDANCE_SCALE,
        ).images
        return list(images)
    except RuntimeError as exc:
        if _is_cuda_oom(exc) and len(prompts_batch) > 1:
            _clear_cuda()
            mid = len(prompts_batch) // 2
            log_event(f"CUDA OOM in generation; splitting batch {len(prompts_batch)} -> {mid}+{len(prompts_batch)-mid}")
            left = generate_images_batched(pipe, prompts_batch[:mid], seeds_batch[:mid], device=device, dtype=dtype)
            right = generate_images_batched(pipe, prompts_batch[mid:], seeds_batch[mid:], device=device, dtype=dtype)
            return left + right
        raise


@torch.no_grad()
def precompute_fp_cache(pipe, prompts: Sequence[str], seeds: Sequence[int], image_pbar=None) -> Dict[Tuple[str, int, int, float], Any]:
    cache: Dict[Tuple[str, int, int, float], Any] = {}
    pending = [(prompt, int(seed), fp_cache_key(prompt, int(seed))) for prompt, seed in eval_pairs(prompts, seeds)]
    gen_batch = resolve_generation_batch_size(len(pending), label="FP cache")

    for b0 in range(0, len(pending), gen_batch):
        chunk = pending[b0 : b0 + gen_batch]
        chunk_prompts = [x[0] for x in chunk]
        chunk_seeds = [x[1] for x in chunk]
        if image_pbar is not None:
            image_pbar.set_postfix_str(f"FP batch={len(chunk)}, seed={chunk_seeds[0]}")
        t0 = time.perf_counter()
        images = generate_images_batched(pipe, chunk_prompts, chunk_seeds)
        synchronize_cuda()
        log_event(f"FP cache batch | n={len(chunk)}, gen={time.perf_counter() - t0:.2f}s, batch={gen_batch}")
        for (_, _, key), image in zip(chunk, images):
            cache[key] = image
        if image_pbar is not None:
            image_pbar.update(len(chunk))
    return cache


def _forward_metric_timesteps(n_steps: int, pipe) -> List[int]:
    n_steps = max(1, int(n_steps))
    scheduler_config = getattr(getattr(pipe, "scheduler", None), "config", None)
    total = int(getattr(scheduler_config, "num_train_timesteps", 1000))
    hi = max(1, total - max(1, total // 10))
    lo = max(1, total // 10)
    if n_steps == 1:
        return [int((hi + lo) // 2)]
    return [int(round(x)) for x in np.linspace(hi, lo, n_steps)]


def _compare_tensors(prefix: str, fp: torch.Tensor, q: torch.Tensor) -> Dict[str, float]:
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
def _make_forward_probe_inputs(pipe, prompts: Sequence[str]):
    device = torch.device(RUNTIME_DEVICE)
    dtype = RUNTIME_DTYPE if device.type == "cuda" else torch.float32
    probe_prompts = list(prompts)[: max(1, int(FORWARD_METRIC_PROMPTS))] or [""]

    pipe.unet.to(device, dtype=dtype)
    pipe.text_encoder.to(device, dtype=dtype)
    text_inputs = pipe.tokenizer(
        probe_prompts,
        padding="max_length",
        max_length=pipe.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    input_ids = text_inputs.input_ids.to(device)
    encoder_hidden_states = pipe.text_encoder(input_ids)[0].to(device=device, dtype=dtype)

    sample_size = getattr(pipe.unet.config, "sample_size", 64)
    if isinstance(sample_size, (tuple, list)):
        height, width = int(sample_size[0]), int(sample_size[1])
    else:
        height = width = int(sample_size)
    generator = torch.Generator(device=device).manual_seed(int(FORWARD_METRIC_SEED))
    sample = torch.randn(
        len(probe_prompts),
        int(pipe.unet.config.in_channels),
        height,
        width,
        generator=generator,
        device=device,
        dtype=dtype,
    )
    timesteps = _forward_metric_timesteps(FORWARD_METRIC_TIMESTEPS, pipe)
    return sample, timesteps, encoder_hidden_states


@torch.no_grad()
def _run_unet_forward_probe(pipe, sample: torch.Tensor, timesteps: Sequence[int], encoder_hidden_states: torch.Tensor):
    outputs: List[torch.Tensor] = []
    mid_outputs: List[torch.Tensor] = []

    def _hook(_module, _inputs, output):
        out = output[0] if isinstance(output, (tuple, list)) else output
        mid_outputs.append(out.detach().float().cpu())

    handle = pipe.unet.mid_block.register_forward_hook(_hook)
    try:
        for timestep in timesteps:
            t = torch.full((sample.shape[0],), int(timestep), device=sample.device, dtype=torch.long)
            out = pipe.unet(sample, t, encoder_hidden_states=encoder_hidden_states).sample
            outputs.append(out.detach().float().cpu())
    finally:
        handle.remove()
    return torch.cat(outputs, dim=0), torch.cat(mid_outputs, dim=0)


@torch.no_grad()
def compute_forward_drift_with_midblock_patch(pipe, mid_block_q: nn.Module, prompts: Sequence[str], eval_label: str) -> Dict[str, float]:
    if not COMPUTE_FORWARD_METRICS or FORWARD_METRIC_PROMPTS <= 0 or FORWARD_METRIC_TIMESTEPS <= 0:
        return {}

    fp_mid_block = pipe.unet.mid_block
    was_training = pipe.unet.training
    try:
        sample, timesteps, encoder_hidden_states = _make_forward_probe_inputs(pipe, prompts)
        fp_out, fp_mid = _run_unet_forward_probe(pipe, sample, timesteps, encoder_hidden_states)

        pipe.unet.mid_block = mid_block_q.to(sample.device, dtype=sample.dtype)
        pipe.unet.eval()
        q_out, q_mid = _run_unet_forward_probe(pipe, sample, timesteps, encoder_hidden_states)

        metrics: Dict[str, float] = {}
        metrics.update(_compare_tensors("unet_forward", fp_out, q_out))
        metrics.update(_compare_tensors("midblock_forward", fp_mid, q_mid))
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
    finally:
        pipe.unet.mid_block = fp_mid_block
        pipe.unet.train(was_training)


@torch.no_grad()
def eval_with_midblock_patch(
    pipe,
    mid_block_q: nn.Module,
    prompts: Sequence[str],
    seeds: Sequence[int],
    fp_cache: Dict[Tuple[str, int, int, float], Any],
    *,
    eval_label: str,
    forward_metrics: Optional[Dict[str, float]],
    metric_objects: Dict[str, Any],
    image_pbar=None,
) -> Tuple[Dict[str, float], List[Dict[str, Any]]]:
    rows: List[Dict[str, Any]] = []
    pairs = eval_pairs(prompts, seeds)
    fp_mid_block = pipe.unet.mid_block
    was_training = pipe.unet.training
    gen_batch = resolve_generation_batch_size(len(pairs), label=eval_label)

    try:
        pipe.unet.mid_block = mid_block_q.to(RUNTIME_DEVICE, dtype=RUNTIME_DTYPE)
        pipe.unet.eval()

        for b0 in range(0, len(pairs), gen_batch):
            chunk = pairs[b0 : b0 + gen_batch]
            chunk_prompts = [prompt for prompt, _ in chunk]
            chunk_seeds = [int(seed) for _, seed in chunk]
            imgs_fp = [fp_cache[fp_cache_key(prompt, seed)] for prompt, seed in chunk]

            if image_pbar is not None:
                image_pbar.set_postfix_str(f"{eval_label} batch={len(chunk)}, seed={chunk_seeds[0]}")
            t_gen0 = time.perf_counter()
            imgs_test = generate_images_batched(pipe, chunk_prompts, chunk_seeds)
            synchronize_cuda()
            t_gen = time.perf_counter() - t_gen0

            if image_pbar is not None:
                image_pbar.update(len(chunk))
                image_pbar.set_postfix_str(f"{eval_label} metrics batch={len(chunk)}")

            t_met0 = time.perf_counter()
            metrics = compute_metrics_batch_against_fp(imgs_fp, imgs_test, chunk_prompts, **metric_objects)
            synchronize_cuda()
            t_met = time.perf_counter() - t_met0
            log_event(
                f"Eval batch | {eval_label} | n={len(chunk)}, gen={t_gen:.2f}s, metrics={t_met:.2f}s, gen_batch={gen_batch}"
            )

            for (prompt, seed), res in zip(chunk, metrics):
                rows.append(
                    {
                        "prompt": prompt,
                        "seed": int(seed),
                        "lpips": float(res["lpips"]),
                        "clip_ti_fp": float(res["clip_ti_fp"]),
                        "clip_ti": float(res["clip_ti"]),
                        "delta_clip_ti": float(res["delta_clip_ti"]),
                        "clip_ii": float(res["clip_ii"]),
                        "pixel_mse": float(res["pixel_mse"]),
                        "pixel_mae": float(res["pixel_mae"]),
                        "psnr": float(res["psnr"]),
                        "ssim": float(res["ssim"]),
                        "hps": float("nan"),
                        "hps_fp": float("nan"),
                    }
                )
    finally:
        pipe.unet.mid_block = fp_mid_block
        pipe.unet.train(was_training)

    def mean(key: str) -> float:
        values = [float(row[key]) for row in rows if math.isfinite(float(row[key]))]
        return float(sum(values) / len(values)) if values else float("nan")

    agg = {
        "lpips": mean("lpips"),
        "clip_ti_fp": mean("clip_ti_fp"),
        "clip_ti": mean("clip_ti"),
        "delta_clip_ti": mean("delta_clip_ti"),
        "clip_ii": mean("clip_ii"),
        "pixel_mse": mean("pixel_mse"),
        "pixel_mae": mean("pixel_mae"),
        "psnr": mean("psnr"),
        "ssim": mean("ssim"),
        "hps": float("nan"),
        "hps_fp": float("nan"),
        "n": len(rows),
    }
    agg.update(dict(forward_metrics or {}))
    return agg, rows


SUMMARY_FIELDS = [
    "label",
    "mode",
    "target_bpp",
    "weight_bits",
    "allowed_bits",
    "bit_histogram",
    "bit_allocation",
    "effective_avg_weight_bits_without_scales",
    "allocation_status",
    "allocation_policy",
    "balanced_integer",
    "semantic_protect",
    "semantic_min_bits",
    "budget_margin_bits",
    "activation_bits",
    "activation_scheme",
    "target_mode",
    "include_conv",
    "target_layers",
    "target_weight_params",
    "target_bias_params",
    "target_weight_fp_bits",
    "raw_weight_bits",
    "quant_weight_bits",
    "quant_weight_bpp",
    "quant_weight_compression",
    "scale_bits",
    "timestep_bucket_weights",
    "clip_ratio_mean",
    "clip_ratio_min",
    "clip_ratio_max",
    "eval_prompts",
    "eval_seeds",
    "eval_items",
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
    "hps",
    "hps_fp",
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
    "calibration_s",
    "candidate_build_s",
    "quant_build_s",
    "eval_s",
    "total_s",
    "total_with_calibration_s",
]


def write_summary_csv(results: Sequence[Dict[str, Any]], path: Path):
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        for result in results:
            row = {field: result.get(field) for field in SUMMARY_FIELDS}
            writer.writerow(row)


def write_rows_jsonl(all_rows: Sequence[Dict[str, Any]], path: Path):
    with path.open("w", encoding="utf-8") as f:
        for row in all_rows:
            f.write(json.dumps(row, ensure_ascii=False, default=_json_default) + "\n")


def save_outputs(results: Sequence[Dict[str, Any]], rows: Sequence[Dict[str, Any]], metadata: Dict[str, Any]):
    write_summary_csv(results, OUTPUT_DIR / "bitsfusion_lite_summary.csv")
    write_rows_jsonl(rows, OUTPUT_DIR / "bitsfusion_lite_rows.jsonl")
    payload = {"metadata": metadata, "results": list(results)}
    with (OUTPUT_DIR / "bitsfusion_lite_results.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, default=_json_default)


def aggregate_clip_ratio_stats(compression: Dict[str, Any]) -> Dict[str, float]:
    per_layer = compression.get("bitsfusion_calibration", {}).get("per_layer", [])
    usable = [item for item in per_layer if not item.get("fallback_reason")]
    if not usable:
        return {
            "clip_ratio_mean": float("nan"),
            "clip_ratio_min": float("nan"),
            "clip_ratio_max": float("nan"),
        }
    total = sum(max(int(item.get("out_features", 0)), 1) for item in usable)
    mean = sum(float(item.get("clip_ratio_mean", 1.0)) * max(int(item.get("out_features", 0)), 1) for item in usable) / max(total, 1)
    return {
        "clip_ratio_mean": float(mean),
        "clip_ratio_min": float(min(float(item.get("clip_ratio_min", 1.0)) for item in usable)),
        "clip_ratio_max": float(max(float(item.get("clip_ratio_max", 1.0)) for item in usable)),
    }


def make_variant_configs() -> List[BitsFusionLiteConfig]:
    variants: List[BitsFusionLiteConfig] = []
    allowed = tuple(sorted(int(x) for x in LAYER_BITS_LIST))
    for target_bpp in TARGET_BPPS:
        variants.append(
            BitsFusionLiteConfig(
                target_bpp=float(target_bpp),
                allowed_bits=allowed,
                activation_bits=int(ACTIVATION_BITS),
                activation_scheme=ACTIVATION_SCHEME,
            )
        )
    return variants


def main():
    log_section("BitsFusion-lite baseline start")
    log_event(f"Output directory: {OUTPUT_DIR}")
    log_event(
        f"device={RUNTIME_DEVICE}, quant_device={QUANT_DEVICE}, dtype={RUNTIME_DTYPE}, metric_device={METRIC_DEVICE}, "
        f"steps={NUM_INFERENCE_STEPS}, guidance={GUIDANCE_SCALE}, seed={GLOBAL_SEED}"
    )
    log_event(
        f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '') or '<unset>'}, "
        f"cuda_count={torch.cuda.device_count() if torch.cuda.is_available() else 0}, "
        f"cuda_devices={_cuda_device_summaries()}"
    )
    log_event(
        f"target_bpps={TARGET_BPPS}, layer_bits={LAYER_BITS_LIST}, activation_bits={ACTIVATION_BITS}, "
        f"activation_scheme={ACTIVATION_SCHEME}, target_mode={TARGET_MODE}, include_conv={INCLUDE_CONV}, "
        f"allocation_policy={ALLOCATION_POLICY}, balanced_integer={BALANCED_INTEGER}, "
        f"semantic_protect={SEMANTIC_PROTECT}, semantic_min_bits={SEMANTIC_MIN_BITS}, "
        f"eval_prompts={EVAL_PROMPT_COUNT}, seeds={EVAL_SEEDS}"
    )
    log_event(
        f"calib_prompts={CALIB_PROMPT_COUNT}, calib_seeds={CALIB_SEEDS}, calib_steps={CALIB_STEPS}, "
        f"calib_max_rows={CALIB_MAX_ROWS_PER_LAYER}, clip_ratio_grid={CLIP_RATIO_GRID}, "
        f"timestep_bucket_weights={_timestep_bucket_weights_dict()}, out_chunk={CLIP_OPT_OUT_CHUNK}"
    )

    log_section("Load SD1.5 and metric models")
    log_cuda_memory("before load")
    pipe = StableDiffusionPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=RUNTIME_DTYPE,
        safety_checker=None,
    ).to(RUNTIME_DEVICE)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.set_progress_bar_config(disable=True)
    optimize_pipeline_for_inference(pipe, "FP pipeline")
    log_cuda_memory("after SD load")

    clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
        "ViT-H-14",
        pretrained="laion2b_s32b_b79k",
    )
    clip_model = clip_model.to(METRIC_DEVICE).eval()
    clip_tokenizer = open_clip.get_tokenizer("ViT-H-14")
    lpips_model = lpips.LPIPS(net="alex").to(METRIC_DEVICE).eval()
    lpips_transform = T.Compose([T.Resize((256, 256)), T.CenterCrop((256, 256)), T.ToTensor()])
    metric_objects = {
        "clip_model": clip_model,
        "clip_tokenizer": clip_tokenizer,
        "clip_preprocess": clip_preprocess,
        "lpips_model": lpips_model,
        "lpips_transform": lpips_transform,
    }
    log_cuda_memory("after metric load")

    log_section("Prompts and target layers")
    prompts = load_pickscore_prompts(EVAL_PROMPT_COUNT, seed=0, eval_mode=True)
    calib_prompts = load_pickscore_prompts(CALIB_PROMPT_COUNT, seed=1, eval_mode=False)
    target_layers = select_target_layers(pipe.unet)
    target_names = list(target_layers.keys())
    print("Selected BitsFusion-lite target layers:")
    for name, layer in target_layers.items():
        print(f"  {name:60s} {type(layer).__name__:8s} weight={tuple(layer.weight.shape)} params={layer.weight.numel():,}")

    metadata: Dict[str, Any] = {
        "script": str(Path(__file__).resolve()),
        "output_dir": str(OUTPUT_DIR),
        "model_id": MODEL_ID,
        "device": RUNTIME_DEVICE,
        "quant_device": QUANT_DEVICE,
        "dtype": str(RUNTIME_DTYPE),
        "metric_device": METRIC_DEVICE,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "cuda_devices": _cuda_device_summaries(),
        "num_inference_steps": NUM_INFERENCE_STEPS,
        "guidance_scale": GUIDANCE_SCALE,
        "global_seed": GLOBAL_SEED,
        "target_mode": TARGET_MODE,
        "include_conv": INCLUDE_CONV,
        "target_names": target_names,
        "eval_prompt_count": len(prompts),
        "eval_seeds": list(EVAL_SEEDS),
        "target_bpps": list(float(x) for x in TARGET_BPPS),
        "layer_bits": list(int(x) for x in LAYER_BITS_LIST),
        "activation_bits": int(ACTIVATION_BITS),
        "activation_scheme": ACTIVATION_SCHEME,
        "allocation_policy": ALLOCATION_POLICY,
        "balanced_integer": bool(BALANCED_INTEGER),
        "semantic_protect": bool(SEMANTIC_PROTECT),
        "semantic_min_bits": int(SEMANTIC_MIN_BITS),
        "sensitivity_multipliers": {
            "cross_attention": float(CROSS_ATTN_SENSITIVITY),
            "self_attention": float(SELF_ATTN_SENSITIVITY),
            "ffn": float(FFN_SENSITIVITY),
            "conv": float(CONV_SENSITIVITY),
        },
        "calib_prompt_count": len(calib_prompts),
        "calib_seeds": list(CALIB_SEEDS),
        "calib_steps": int(CALIB_STEPS),
        "timestep_bucket_weights": _timestep_bucket_weights_dict(),
        "calib_max_rows_per_layer": int(CALIB_MAX_ROWS_PER_LAYER),
        "clip_ratio_grid": list(float(x) for x in CLIP_RATIO_GRID),
        "clip_opt_out_chunk": int(CLIP_OPT_OUT_CHUNK),
    }
    with (OUTPUT_DIR / "bitsfusion_lite_config.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False, default=_json_default)

    total_images = len(prompts) * len(EVAL_SEEDS) * (1 + len(make_variant_configs()))
    image_pbar = tqdm(total=total_images, desc="BitsFusion-lite images", unit="img")
    try:
        log_section("BitsFusion-lite timestep calibration")
        t_calib0 = time.perf_counter()
        calibration_rows, calibration_bucket_rows, calibration_bucket_counts = collect_bitsfusion_calibration_rows(
            pipe,
            target_layers,
            calib_prompts,
            CALIB_SEEDS,
        )
        synchronize_cuda()
        t_calib = time.perf_counter() - t_calib0
        metadata["calibration_s"] = float(t_calib)
        metadata["calibration_rows"] = {
            name: {"rows": int(rows.shape[0]), "dim": int(rows.shape[1])} for name, rows in calibration_rows.items()
        }
        metadata["calibration_timestep_bucket_rows"] = {
            name: {
                bucket_name: int(bucket_rows.shape[0])
                for bucket_name, bucket_rows in buckets.items()
            }
            for name, buckets in calibration_bucket_rows.items()
        }
        metadata["calibration_timestep_bucket_raw_counts"] = calibration_bucket_counts
        if not calibration_rows:
            raise RuntimeError("BITSFUSION calibration collected no activation rows; refusing to run scalar fallback silently.")
        missing_calibration = [name for name in target_names if name not in calibration_rows]
        if missing_calibration:
            log_event("BITSFUSION calibration missing layers: " + ", ".join(missing_calibration))
        save_outputs([], [], metadata)
        log_event(f"BITSFUSION calibration complete | layers={len(calibration_rows)}, time={t_calib:.2f}s")

        log_section("FP image cache")
        fp_cache = precompute_fp_cache(pipe, prompts, EVAL_SEEDS, image_pbar=image_pbar)

        results: List[Dict[str, Any]] = []
        all_rows: List[Dict[str, Any]] = []
        variants = make_variant_configs()

        log_section("BitsFusion-lite candidate table")
        t_candidate0 = time.perf_counter()
        candidate_table = build_bitsfusion_candidate_table(
            target_layers,
            variants[0],
            calibration_rows=calibration_rows,
            calibration_bucket_rows=calibration_bucket_rows,
            quant_device=QUANT_DEVICE,
        )
        synchronize_cuda(QUANT_DEVICE)
        t_candidate = time.perf_counter() - t_candidate0
        metadata["candidate_build_s"] = float(t_candidate)
        metadata["candidate_layer_bits"] = list(int(x) for x in variants[0].allowed_bits)
        save_outputs(results, all_rows, metadata)
        log_event(
            f"BitsFusion candidate table complete | layers={len(candidate_table)}, "
            f"bits={variants[0].allowed_bits}, time={t_candidate:.2f}s"
        )

        log_section("BitsFusion-lite variants")
        for variant_idx, cfg in enumerate(variants, start=1):
            label = cfg.label
            log_section(f"[{variant_idx}/{len(variants)}] {label}")
            set_global_seed(GLOBAL_SEED + variant_idx)
            t_variant0 = time.perf_counter()

            t_quant0 = time.perf_counter()
            mid_block_q, compression = build_bitsfusion_lite_midblock(
                pipe,
                target_layers,
                cfg,
                calibration_rows=calibration_rows,
                calibration_bucket_rows=calibration_bucket_rows,
                candidate_table=candidate_table,
                quant_device=QUANT_DEVICE,
            )
            synchronize_cuda(QUANT_DEVICE)
            t_patch = time.perf_counter() - t_quant0
            t_quant = t_candidate + t_patch
            totals = compression["totals"]
            allocation = compression.get("allocation", {})
            clip_summary = aggregate_clip_ratio_stats(compression)
            log_event(
                f"Built {label}: target_layers={totals['target_layers']}, "
                f"target_bpp={cfg.target_bpp:.3f}, actual_bpp={totals['quant_weight_bpp']:.4f}, "
                f"compression={totals['quant_weight_compression']:.3f}x, "
                f"bits={allocation.get('bit_histogram', {})}, status={allocation.get('status')}, "
                f"clip_mean={clip_summary['clip_ratio_mean']:.3f}, "
                f"candidate_build={t_candidate:.2f}s, patch_build={t_patch:.2f}s"
            )

            forward_metrics = compute_forward_drift_with_midblock_patch(pipe, mid_block_q, prompts, eval_label=label)

            t_eval0 = time.perf_counter()
            agg, rows = eval_with_midblock_patch(
                pipe,
                mid_block_q,
                prompts,
                EVAL_SEEDS,
                fp_cache,
                eval_label=label,
                forward_metrics=forward_metrics,
                metric_objects=metric_objects,
                image_pbar=image_pbar,
            )
            synchronize_cuda()
            t_eval = time.perf_counter() - t_eval0
            t_variant = time.perf_counter() - t_variant0
            t_total = t_candidate + t_variant

            result = {
                "label": label,
                "mode": cfg.mode,
                "target_bpp": float(cfg.target_bpp),
                "weight_bits": "mixed",
                "allowed_bits": ",".join(str(int(x)) for x in cfg.allowed_bits),
                "bit_histogram": json.dumps(allocation.get("bit_histogram", {}), sort_keys=True),
                "bit_allocation": json.dumps(allocation.get("bit_allocation", {}), sort_keys=True),
                "effective_avg_weight_bits_without_scales": float(
                    allocation.get("effective_avg_weight_bits_without_scales", float("nan"))
                ),
                "allocation_status": str(allocation.get("status", "")),
                "allocation_policy": str(allocation.get("allocation_policy", ALLOCATION_POLICY)),
                "balanced_integer": bool(BALANCED_INTEGER),
                "semantic_protect": bool(allocation.get("semantic_protect", SEMANTIC_PROTECT)),
                "semantic_min_bits": int(allocation.get("semantic_min_bits", SEMANTIC_MIN_BITS)),
                "budget_margin_bits": float(allocation.get("budget_margin_bits", float("nan"))),
                "activation_bits": int(cfg.activation_bits),
                "activation_scheme": cfg.activation_scheme,
                "target_mode": TARGET_MODE,
                "include_conv": bool(INCLUDE_CONV),
                "target_layers": int(totals["target_layers"]),
                "target_weight_params": int(totals["target_weight_params"]),
                "target_bias_params": int(totals["target_bias_params"]),
                "target_weight_fp_bits": int(totals["target_weight_fp_bits"]),
                "raw_weight_bits": int(totals["raw_weight_bits"]),
                "quant_weight_bits": int(totals["quant_weight_bits"]),
                "quant_weight_bpp": float(totals["quant_weight_bpp"]),
                "quant_weight_compression": float(totals["quant_weight_compression"]),
                "scale_bits": int(totals["scale_bits"]),
                "timestep_bucket_weights": json.dumps(_timestep_bucket_weights_dict(), sort_keys=True),
                **clip_summary,
                "eval_prompts": len(prompts),
                "eval_seeds": len(EVAL_SEEDS),
                "eval_items": len(prompts) * len(EVAL_SEEDS),
                "score": float(agg["clip_ii"]) - float(agg["lpips"]),
                **agg,
                "calibration_s": float(t_calib),
                "candidate_build_s": float(t_candidate),
                "quant_build_s": float(t_quant),
                "eval_s": float(t_eval),
                "total_s": float(t_total),
                "total_with_calibration_s": float(t_total + t_calib),
                "compression": compression,
            }
            results.append(result)

            for row in rows:
                row_with_label = {
                    "label": label,
                    "mode": cfg.mode,
                    "target_bpp": float(cfg.target_bpp),
                    "weight_bits": "mixed",
                    "activation_bits": cfg.activation_bits,
                }
                row_with_label.update(row)
                all_rows.append(row_with_label)

            save_outputs(results, all_rows, metadata)
            log_event(
                f"Result | {label} | target_bpp={cfg.target_bpp:.3f} | actual_bpp={result['quant_weight_bpp']:.4f} | "
                f"score={result['score']:.4f} | "
                f"CLIP i-i={result['clip_ii']:.4f} | LPIPS={result['lpips']:.4f} | "
                f"CLIP t-i={result['clip_ti']:.4f} | PSNR={result['psnr']:.2f} | SSIM={result['ssim']:.4f} | "
                f"timing quant={t_quant:.1f}s eval={t_eval:.1f}s total={t_total:.1f}s"
            )

            del mid_block_q
            _clear_cuda()

    finally:
        image_pbar.close()

    log_section("BitsFusion-lite summary")
    for result in results:
        print(
            f"{result['label']:42s} target={result['target_bpp']:.2f} bpp={result['quant_weight_bpp']:.3f} "
            f"score={result['score']:.4f} clip_ii={result['clip_ii']:.4f} lpips={result['lpips']:.4f} "
            f"clip_ti={result['clip_ti']:.4f} psnr={result['psnr']:.2f} "
            f"time={result['total_s']:.1f}s"
        )
    print(f"Saved summary to: {OUTPUT_DIR / 'bitsfusion_lite_summary.csv'}")
    print(f"Saved JSON to   : {OUTPUT_DIR / 'bitsfusion_lite_results.json'}")


if __name__ == "__main__":
    main()


