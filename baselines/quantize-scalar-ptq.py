# -*- coding: utf-8 -*-
"""Scalar PTQ baselines for SD1.5.

Two branches are implemented:
  1. weight_only: per-output-channel symmetric scalar weight quantization.
  2. weight_activation: the same weight quantization plus dynamic fake quantization
     of operator inputs at the quantized Linear/Conv2d layers.

The script intentionally mirrors the RVQ/VQDM evaluation protocol: PickScore
validation prompts, DDIM, fixed seeds, CLIP image-image/text-image, LPIPS,
cheap pixel metrics, forward drift, and per-variant timing.
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
    if not _pre_torch_env_bool("SCALAR_PTQ_SELECT_IDLE_GPUS", True):
        print("SCALAR_PTQ_SELECT_IDLE_GPUS=0; keeping caller GPU visibility.")
        return

    try:
        gpu_state = _query_nvidia_smi_gpu_state()
    except FileNotFoundError:
        message = "nvidia-smi not found; cannot verify idle GPUs before torch import."
        if _pre_torch_env_bool("SCALAR_PTQ_REQUIRE_IDLE_GPUS", True):
            raise SystemExit(message)
        print("WARNING: " + message)
        return
    except Exception as exc:
        message = f"nvidia-smi idle-GPU preselection failed: {type(exc).__name__}: {exc}"
        if _pre_torch_env_bool("SCALAR_PTQ_REQUIRE_IDLE_GPUS", True):
            raise SystemExit(message)
        print("WARNING: " + message)
        return

    if not gpu_state:
        message = "nvidia-smi returned no GPU rows; cannot verify idle GPUs."
        if _pre_torch_env_bool("SCALAR_PTQ_REQUIRE_IDLE_GPUS", True):
            raise SystemExit(message)
        print("WARNING: " + message)
        return

    max_used_mem_mib = _pre_torch_env_int("SCALAR_PTQ_IDLE_GPU_MAX_USED_MEM_MIB", 2048)
    max_devices = _pre_torch_env_int("SCALAR_PTQ_AUTO_MAX_DEVICES", 0)  # 0 = all idle GPUs
    require_idle = _pre_torch_env_bool("SCALAR_PTQ_REQUIRE_IDLE_GPUS", True)
    validate_preset = _pre_torch_env_bool("SCALAR_PTQ_VALIDATE_PRESET_GPUS", True)
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
LOGGER = logging.getLogger("scalar-ptq")


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
    file_handler = logging.FileHandler(output_dir / "scalar_ptq.log", encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s", "%H:%M:%S"))
    LOGGER.addHandler(file_handler)


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
RUN_STAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_ROOT = Path(os.environ.get("SCALAR_PTQ_OUTPUT_ROOT", str(SCRIPT_DIR / "runs"))).expanduser()
OUTPUT_DIR = OUTPUT_ROOT / os.environ.get("SCALAR_PTQ_RUN_NAME", f"scalar_ptq_{RUN_STAMP}")
_add_file_logger(OUTPUT_DIR)


@dataclass(frozen=True)
class ScalarPTQConfig:
    weight_bits: int
    mode: str
    activation_bits: int = 0
    weight_scale_bits: int = 16
    activation_scheme: str = "dynamic_per_sample"

    @property
    def label(self) -> str:
        if self.mode == "weight_activation":
            return f"scalar_ptq_w{self.weight_bits}a{self.activation_bits}"
        return f"scalar_ptq_w{self.weight_bits}"


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


GLOBAL_SEED = _env_int("SCALAR_PTQ_SEED", 20260515)
set_global_seed(GLOBAL_SEED)

torch.backends.cudnn.allow_tf32 = _env_bool("SCALAR_PTQ_ALLOW_TF32", True)
torch.backends.cuda.matmul.allow_tf32 = _env_bool("SCALAR_PTQ_ALLOW_TF32", True)

RUNTIME_DEVICE = os.environ.get("SCALAR_PTQ_RUNTIME_DEVICE") or ("cuda:0" if torch.cuda.is_available() else "cpu")
RUNTIME_DTYPE = torch.float16 if str(RUNTIME_DEVICE).startswith("cuda") else torch.float32
QUANT_DEVICE = os.environ.get("SCALAR_PTQ_QUANT_DEVICE") or RUNTIME_DEVICE
METRIC_DEVICE = os.environ.get("SCALAR_PTQ_METRIC_DEVICE")
if not METRIC_DEVICE:
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        METRIC_DEVICE = f"cuda:{torch.cuda.device_count() - 1}"
    else:
        METRIC_DEVICE = RUNTIME_DEVICE

MODEL_ID = os.environ.get("SCALAR_PTQ_MODEL_ID", "runwayml/stable-diffusion-v1-5")
NUM_INFERENCE_STEPS = _env_int("SCALAR_PTQ_STEPS", 50)
GUIDANCE_SCALE = _env_float("SCALAR_PTQ_GUIDANCE", 5.0)
EVAL_PROMPT_COUNT = _env_int("SCALAR_PTQ_EVAL_PROMPTS", 32)
EVAL_SEEDS = _parse_int_list(os.environ.get("SCALAR_PTQ_EVAL_SEEDS", "1234,4321,999"), min_value=0)
WEIGHT_BITS_LIST = _parse_int_list(os.environ.get("SCALAR_PTQ_WEIGHT_BITS", "4,3,2"), min_value=2)
ACTIVATION_BITS_LIST = _parse_int_list(os.environ.get("SCALAR_PTQ_ACT_BITS", "8"), min_value=2)
ACTIVATION_SCHEME = os.environ.get("SCALAR_PTQ_ACT_SCHEME", "dynamic_per_sample").strip().lower()
BRANCHES = _parse_str_list(os.environ.get("SCALAR_PTQ_BRANCHES", "weight_only,weight_activation"))
TARGET_MODE = os.environ.get("SCALAR_PTQ_TARGET_MODE", "all_midblock").strip().lower()
SINGLE_TARGET_LAYER = os.environ.get(
    "SCALAR_PTQ_SINGLE_TARGET_LAYER",
    "mid_block.attentions.0.transformer_blocks.0.ff.net.0.proj",
)
INCLUDE_CONV = _env_bool("SCALAR_PTQ_INCLUDE_CONV", True)
GEN_BATCH_ENV = _env_int("SCALAR_PTQ_GEN_BATCH", 0)
GEN_BATCH_MAX = _env_int("SCALAR_PTQ_GEN_BATCH_MAX", 0)
GEN_BATCH_MIN = _env_int("SCALAR_PTQ_GEN_BATCH_MIN", 1)
GPU_TARGET_MEM_FRACTION = _env_float("SCALAR_PTQ_GPU_TARGET_MEM_FRACTION", 0.78)
GPU_RESERVE_GIB = _env_float("SCALAR_PTQ_GPU_RESERVE_GIB", 4.0)
GEN_ESTIMATED_GIB_PER_IMAGE = _env_float("SCALAR_PTQ_GEN_ESTIMATED_GIB_PER_IMAGE", 1.8)
COMPUTE_FORWARD_METRICS = _env_bool("SCALAR_PTQ_FORWARD_METRICS", True)
FORWARD_METRIC_PROMPTS = _env_int("SCALAR_PTQ_FORWARD_METRIC_PROMPTS", 2)
FORWARD_METRIC_TIMESTEPS = _env_int("SCALAR_PTQ_FORWARD_METRIC_TIMESTEPS", 3)
FORWARD_METRIC_SEED = _env_int("SCALAR_PTQ_FORWARD_METRIC_SEED", 20260515)
FAST_SMOKE = _env_bool("SCALAR_PTQ_FAST_SMOKE", False)

if FAST_SMOKE:
    EVAL_PROMPT_COUNT = min(EVAL_PROMPT_COUNT, 1)
    EVAL_SEEDS = EVAL_SEEDS[:1]
    WEIGHT_BITS_LIST = WEIGHT_BITS_LIST[:1]
    BRANCHES = BRANCHES[:1]

valid_branches = {"weight_only", "weight_activation"}
unknown_branches = [branch for branch in BRANCHES if branch not in valid_branches]
if unknown_branches:
    raise ValueError(f"Unsupported SCALAR_PTQ_BRANCHES entries: {unknown_branches}; expected {sorted(valid_branches)}")

valid_activation_schemes = {"dynamic_per_tensor", "dynamic_per_sample"}
if ACTIVATION_SCHEME not in valid_activation_schemes:
    raise ValueError(
        f"Unsupported SCALAR_PTQ_ACT_SCHEME={ACTIVATION_SCHEME!r}; "
        f"expected one of {sorted(valid_activation_schemes)}"
    )

valid_target_modes = {"hardcoded_midblock", "all_midblock_linear", "all_midblock", "single_largest_linear", "single_named_linear"}
if TARGET_MODE not in valid_target_modes:
    raise ValueError(f"Unsupported SCALAR_PTQ_TARGET_MODE={TARGET_MODE!r}; expected one of {sorted(valid_target_modes)}")


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


def _resolve_pickscore_csv(filename: str, env_name: str = "SCALAR_PTQ_PICKSCORE_DATA_DIR") -> Path:
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


def _symmetric_quant_bounds(bits: int) -> Tuple[int, int]:
    qmax = (1 << (int(bits) - 1)) - 1
    qmin = -qmax
    return qmin, qmax


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

    qmin, qmax = _symmetric_quant_bounds(bits)
    reduce_dims = tuple(range(1, w.ndim))
    max_abs = w.abs().amax(dim=reduce_dims, keepdim=True).clamp_min(1e-12)
    scales = max_abs / float(qmax)
    q = torch.round(w / scales).clamp(qmin, qmax)
    return (q * scales).float(), scales.float()


class ScalarQuantLinear(nn.Module):
    def __init__(self, source: nn.Linear, cfg: ScalarPTQConfig, *, quant_device: str | torch.device):
        super().__init__()
        q_device = torch.device(quant_device)
        q_weight, scales = quantize_weight_per_output_channel(source.weight, cfg.weight_bits, quant_device=q_device)
        self.register_buffer("weight", q_weight)
        self.register_buffer("weight_scales", scales.reshape(-1).contiguous())
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


class ScalarQuantConv2d(nn.Module):
    def __init__(self, source: nn.Conv2d, cfg: ScalarPTQConfig, *, quant_device: str | torch.device):
        super().__init__()
        q_device = torch.device(quant_device)
        q_weight, scales = quantize_weight_per_output_channel(source.weight, cfg.weight_bits, quant_device=q_device)
        self.register_buffer("weight", q_weight)
        self.register_buffer("weight_scales", scales.reshape(-1).contiguous())
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


def scalar_compression_stats(
    target_layers: Dict[str, nn.Module],
    cfg: ScalarPTQConfig,
    *,
    fp_bits: int = 16,
) -> Dict[str, Any]:
    total_weight_params = 0
    total_bias_params = 0
    total_raw_weight_bits = 0
    total_quant_weight_bits = 0
    total_scale_bits = 0
    per_layer: List[Dict[str, Any]] = []

    for name, layer in target_layers.items():
        weight_params = int(layer.weight.numel())
        bias_params = int(layer.bias.numel()) if getattr(layer, "bias", None) is not None else 0
        scale_count = int(layer.weight.shape[0])
        raw_weight_bits = weight_params * int(cfg.weight_bits)
        scale_bits = scale_count * int(cfg.weight_scale_bits)
        quant_weight_bits = raw_weight_bits + scale_bits

        total_weight_params += weight_params
        total_bias_params += bias_params
        total_raw_weight_bits += raw_weight_bits
        total_quant_weight_bits += quant_weight_bits
        total_scale_bits += scale_bits

        per_layer.append(
            {
                "name": name,
                "layer_type": type(layer).__name__,
                "weight_shape": list(layer.weight.shape),
                "weight_params": weight_params,
                "bias_params": bias_params,
                "weight_bits": int(cfg.weight_bits),
                "activation_bits": int(cfg.activation_bits),
                "scale_count": scale_count,
                "scale_bits": scale_bits,
                "quant_weight_bits": quant_weight_bits,
                "weight_bpp_with_scales": quant_weight_bits / max(weight_params, 1),
            }
        )

    fp_weight_bits = total_weight_params * int(fp_bits)
    return {
        "config": {
            "mode": cfg.mode,
            "weight_bits": int(cfg.weight_bits),
            "activation_bits": int(cfg.activation_bits),
            "weight_scale_bits": int(cfg.weight_scale_bits),
            "activation_scheme": cfg.activation_scheme,
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
def build_scalar_quant_midblock(
    pipe_fp: StableDiffusionPipeline,
    target_layers: Dict[str, nn.Module],
    cfg: ScalarPTQConfig,
    *,
    quant_device: str | torch.device = QUANT_DEVICE,
) -> Tuple[nn.Module, Dict[str, Any]]:
    q_device = torch.device(quant_device)
    mid_block_q = copy.deepcopy(pipe_fp.unet.mid_block).to(q_device, dtype=torch.float32)
    stats = scalar_compression_stats(target_layers, cfg)

    for full_name, source_layer in target_layers.items():
        if isinstance(source_layer, nn.Linear):
            q_layer = ScalarQuantLinear(source_layer, cfg, quant_device=q_device).to(q_device, dtype=torch.float32)
        elif isinstance(source_layer, nn.Conv2d):
            q_layer = ScalarQuantConv2d(source_layer, cfg, quant_device=q_device).to(q_device, dtype=torch.float32)
        else:
            raise TypeError(f"Unsupported target layer type for {full_name}: {type(source_layer).__name__}")
        replace_midblock_submodule(mid_block_q, full_name, q_layer)

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
    "weight_bits",
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
    "quant_build_s",
    "eval_s",
    "total_s",
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
    write_summary_csv(results, OUTPUT_DIR / "scalar_ptq_summary.csv")
    write_rows_jsonl(rows, OUTPUT_DIR / "scalar_ptq_rows.jsonl")
    payload = {"metadata": metadata, "results": list(results)}
    with (OUTPUT_DIR / "scalar_ptq_results.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, default=_json_default)


def make_variant_configs() -> List[ScalarPTQConfig]:
    variants: List[ScalarPTQConfig] = []
    for branch in BRANCHES:
        for weight_bits in WEIGHT_BITS_LIST:
            if branch == "weight_only":
                variants.append(ScalarPTQConfig(weight_bits=int(weight_bits), mode="weight_only", activation_bits=0))
            else:
                for activation_bits in ACTIVATION_BITS_LIST:
                    variants.append(
                        ScalarPTQConfig(
                            weight_bits=int(weight_bits),
                            mode="weight_activation",
                            activation_bits=int(activation_bits),
                            activation_scheme=ACTIVATION_SCHEME,
                        )
                    )
    return variants


def main():
    log_section("Scalar PTQ baseline start")
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
        f"branches={BRANCHES}, weight_bits={WEIGHT_BITS_LIST}, activation_bits={ACTIVATION_BITS_LIST}, "
        f"activation_scheme={ACTIVATION_SCHEME}, target_mode={TARGET_MODE}, include_conv={INCLUDE_CONV}, "
        f"eval_prompts={EVAL_PROMPT_COUNT}, seeds={EVAL_SEEDS}"
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
    target_layers = select_target_layers(pipe.unet)
    target_names = list(target_layers.keys())
    print("Selected scalar PTQ target layers:")
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
        "branches": list(BRANCHES),
        "weight_bits": list(WEIGHT_BITS_LIST),
        "activation_bits": list(ACTIVATION_BITS_LIST),
        "activation_scheme": ACTIVATION_SCHEME,
    }
    with (OUTPUT_DIR / "scalar_ptq_config.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False, default=_json_default)

    total_images = len(prompts) * len(EVAL_SEEDS) * (1 + len(make_variant_configs()))
    image_pbar = tqdm(total=total_images, desc="scalar PTQ images", unit="img")
    try:
        log_section("FP image cache")
        fp_cache = precompute_fp_cache(pipe, prompts, EVAL_SEEDS, image_pbar=image_pbar)

        results: List[Dict[str, Any]] = []
        all_rows: List[Dict[str, Any]] = []
        variants = make_variant_configs()
        log_section("Scalar PTQ variants")
        for variant_idx, cfg in enumerate(variants, start=1):
            label = cfg.label
            log_section(f"[{variant_idx}/{len(variants)}] {label}")
            set_global_seed(GLOBAL_SEED + variant_idx)
            t_variant0 = time.perf_counter()

            t_quant0 = time.perf_counter()
            mid_block_q, compression = build_scalar_quant_midblock(pipe, target_layers, cfg, quant_device=QUANT_DEVICE)
            synchronize_cuda(QUANT_DEVICE)
            t_quant = time.perf_counter() - t_quant0
            totals = compression["totals"]
            log_event(
                f"Built {label}: target_layers={totals['target_layers']}, "
                f"bpp={totals['quant_weight_bpp']:.4f}, compression={totals['quant_weight_compression']:.3f}x, "
                f"quant_build={t_quant:.2f}s"
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
            t_total = time.perf_counter() - t_variant0

            result = {
                "label": label,
                "mode": cfg.mode,
                "weight_bits": int(cfg.weight_bits),
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
                "eval_prompts": len(prompts),
                "eval_seeds": len(EVAL_SEEDS),
                "eval_items": len(prompts) * len(EVAL_SEEDS),
                **agg,
                "score": float(agg["clip_ii"]) - float(agg["lpips"]),
                "quant_build_s": float(t_quant),
                "eval_s": float(t_eval),
                "total_s": float(t_total),
                "compression": compression,
            }
            results.append(result)

            for row in rows:
                row_with_label = {"label": label, "mode": cfg.mode, "weight_bits": cfg.weight_bits, "activation_bits": cfg.activation_bits}
                row_with_label.update(row)
                all_rows.append(row_with_label)

            save_outputs(results, all_rows, metadata)
            log_event(
                f"Result | {label} | bpp={result['quant_weight_bpp']:.4f} | "
                f"CLIP i-i={result['clip_ii']:.4f} | LPIPS={result['lpips']:.4f} | "
                f"CLIP t-i={result['clip_ti']:.4f} | PSNR={result['psnr']:.2f} | SSIM={result['ssim']:.4f} | "
                f"timing quant={t_quant:.1f}s eval={t_eval:.1f}s total={t_total:.1f}s"
            )

            del mid_block_q
            _clear_cuda()

    finally:
        image_pbar.close()

    log_section("Scalar PTQ summary")
    for result in results:
        print(
            f"{result['label']:24s} bpp={result['quant_weight_bpp']:.3f} "
            f"clip_ii={result['clip_ii']:.4f} lpips={result['lpips']:.4f} "
            f"clip_ti={result['clip_ti']:.4f} psnr={result['psnr']:.2f} "
            f"time={result['total_s']:.1f}s"
        )
    print(f"Saved summary to: {OUTPUT_DIR / 'scalar_ptq_summary.csv'}")
    print(f"Saved JSON to   : {OUTPUT_DIR / 'scalar_ptq_results.json'}")


if __name__ == "__main__":
    main()
