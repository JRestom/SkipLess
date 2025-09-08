#!/usr/bin/env python3
import os, time, json, yaml, statistics, torch
from typing import Dict, Any

torch.backends.cudnn.benchmark = True  # ok for GPU inference; ignored on CPU

import torch.nn as nn

class PrunableConvNeXtBlock(nn.Module):
    """
    Minimal shim so torch.load() can unpickle models saved with this wrapper.
    For benchmarking, just call through to the wrapped block.
    """
    def __init__(self, block: nn.Module):
        super().__init__()
        self.block = block
        # attribute may exist in saved models; keep for compatibility
        self.disable_main = False

    def forward(self, x):
        # During inference/benchmark we just run the wrapped block.
        # (In your pruning script, pruned ones were finalized to nn.Identity.)
        return self.block(x)

# ------------------------- IO -------------------------
def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)

# ----------------------- Models -----------------------
def get_model_by_name(name: str, num_classes: int = 1000):
    """
    Try your local 'models' first; fall back to torchvision registry.
    Extend as needed.
    """
    try:
        from models import resnet50, resnet101  # your local defs
        local = {
            "resnet50": lambda: resnet50(num_classes=num_classes),
            "resnet101": lambda: resnet101(num_classes=num_classes),
        }
        if name in local:
            return local[name]()
    except Exception:
        pass

    # torchvision fallback
    try:
        from torchvision.models import get_model
        return get_model(name, weights=None)
    except Exception as e:
        raise ValueError(f"Unknown model '{name}' and torchvision fallback failed: {e}")

def smart_load_model(cfg: Dict[str, Any]) -> torch.nn.Module:
    """
    Loads either:
      - a full checkpoint dict with 'model' (your *_full.pt), or
      - a pure state_dict (weights only), requiring 'baseline_model' to rebuild arch.
    """
    device = cfg["device"]
    pruned_path = cfg.get("pruned_model_path")
    baseline_name = cfg.get("baseline_model")
    num_classes = int(cfg.get("num_classes", 1000))

    if pruned_path:
        obj = torch.load(pruned_path, map_location=device)
        # Full object save: {"model": model, "cfg": ..., "pruned_blocks": [...]}
        if isinstance(obj, dict) and "model" in obj:
            model = obj["model"].to(device).eval()
            return model
        # state_dict path
        if baseline_name is None:
            raise RuntimeError("state_dict provided but 'baseline_model' missing in config.")
        model = get_model_by_name(baseline_name, num_classes=num_classes)
        model.load_state_dict(obj)
        return model.to(device).eval()

    # Baseline-by-name only
    if not baseline_name:
        raise RuntimeError("Provide either 'pruned_model_path' or 'baseline_model' in config.")
    model = get_model_by_name(baseline_name, num_classes=num_classes)
    return model.to(device).eval()

# -------------------- Timing helpers -------------------
def _cuda_sync_if_needed(device: str):
    if str(device).startswith("cuda"):
        torch.cuda.synchronize()

def _time_gpu_forward(model, inp, num_batches: int, amp: bool) -> float:
    """Accurate GPU timing via CUDA events (seconds)."""
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    _cuda_sync_if_needed("cuda")
    starter.record()
    ctx = (torch.autocast(device_type="cuda", dtype=torch.float16) if amp else torch.no_grad())
    with ctx:
        for _ in range(num_batches):
            _ = model(inp)
    ender.record()
    torch.cuda.synchronize()
    ms = starter.elapsed_time(ender)
    return ms / 1000.0

def _time_cpu_forward(model, inp, num_batches: int) -> float:
    """High-res CPU timing (seconds)."""
    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(num_batches):
            _ = model(inp)
    end = time.perf_counter()
    return end - start

# -------------------- Core benchmark -------------------
def measure_throughput(model: torch.nn.Module, cfg: Dict[str, Any]):
    device = cfg["device"]
    bs = int(cfg["batch_size"])
    nb = int(cfg["num_batches"])
    warmup = int(cfg["warmup"])
    amp = bool(cfg.get("amp", False))
    channels_last = bool(cfg.get("channels_last", False))
    force_fp16 = bool(cfg.get("force_fp16", False))  # force input dtype to fp16
    dtype = torch.float16 if (force_fp16 and str(device).startswith("cuda")) else torch.float32

    # Optional CPU tuning
    if device == "cpu" and cfg.get("cpu_threads"):
        torch.set_num_threads(int(cfg["cpu_threads"]))
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision(cfg.get("float32_matmul_precision", "high"))

    model.eval().to(device)
    if channels_last and str(device).startswith("cuda"):
        model.to(memory_format=torch.channels_last)

    inp = torch.randn(bs, 3, 224, 224, device=device, dtype=dtype)
    if channels_last and str(device).startswith("cuda"):
        inp = inp.contiguous(memory_format=torch.channels_last)

    # Warmup
    with (torch.autocast(device_type="cuda", dtype=torch.float16) if (amp and str(device).startswith("cuda")) else torch.no_grad()):
        for _ in range(warmup):
            _ = model(inp)
    _cuda_sync_if_needed(device)

    # Timed run
    if str(device).startswith("cuda"):
        elapsed = _time_gpu_forward(model, inp, nb, amp)
    else:
        elapsed = _time_cpu_forward(model, inp, nb)

    throughput = (bs * nb) / elapsed
    lat_batch = elapsed / nb
    lat_image = lat_batch / bs
    return throughput, lat_batch, lat_image

def benchmark(config_path: str):
    cfg = load_config(config_path)
    device = cfg["device"]

    # Load target model
    model = smart_load_model(cfg)

    # Run repeats
    reps = int(cfg["repeats"])
    thr_list, lb_list, li_list = [], [], []
    name = cfg.get("title", "Model")

    print(f"\n📦 Benchmarking: {name}")
    for i in range(reps):
        thr, lb, li = measure_throughput(model, cfg)
        thr_list.append(thr); lb_list.append(lb); li_list.append(li)
        print(f"  Run {i+1}/{reps}: {thr:.2f} samples/s | {lb*1000:.2f} ms/batch | {li*1000:.2f} ms/image")

    # Summary stats (min, max, mean, median)
    def stats(xs):
        return (min(xs), max(xs), statistics.mean(xs), statistics.median(xs))

    thr_min, thr_max, thr_mean, thr_med = stats(thr_list)
    lb_min, lb_max, lb_mean, lb_med = stats(lb_list)
    li_min, li_max, li_mean, li_med = stats(li_list)

    print(f"\n📊 {name} Summary (device={device}, batch_size={cfg['batch_size']}, "
          f"amp={cfg.get('amp', False)}, channels_last={cfg.get('channels_last', False)}):")
    print(f"  Throughput (samples/s): min {thr_min:.2f} | max {thr_max:.2f} | mean {thr_mean:.2f} | median {thr_med:.2f}")
    print(f"  Latency/batch (ms):     min {lb_min*1000:.2f} | max {lb_max*1000:.2f} | mean {lb_mean*1000:.2f} | median {lb_med*1000:.2f}")
    print(f"  Latency/image (ms):     min {li_min*1000:.2f} | max {li_max*1000:.2f} | mean {li_mean*1000:.2f} | median {li_med*1000:.2f}")

    # Optional dump (now includes mean and median)
    out_json = cfg.get("save_summary_json")
    if out_json:
        os.makedirs(os.path.dirname(out_json), exist_ok=True)
        summary = {
            "title": name,
            "device": device,
            "batch_size": cfg["batch_size"],
            "amp": bool(cfg.get("amp", False)),
            "channels_last": bool(cfg.get("channels_last", False)),
            "repeats": reps,
            "throughput_samples_per_s": thr_list,
            "latency_batch_s": lb_list,
            "latency_image_s": li_list,
            # aggregated stats
            "throughput_mean": thr_mean,
            "throughput_median": thr_med,
            "latency_batch_mean_ms": lb_mean * 1000.0,
            "latency_batch_median_ms": lb_med * 1000.0,
            "latency_image_mean_ms": li_mean * 1000.0,
            "latency_image_median_ms": li_med * 1000.0,
        }
        with open(out_json, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\n💾 Saved summary JSON → {out_json}")

# ------------------------ Main ------------------------
if __name__ == "__main__":
    # single required arg in config; no CLI overrides
    CONFIG_PATH = os.environ.get("BENCH_CONFIG", None)
    if CONFIG_PATH is None:
        # fallback to argv[1] without argparse dependency creep
        import sys
        if len(sys.argv) != 2:
            raise SystemExit("Usage: python benchmark_inference.py <config.yaml>  (all settings come from YAML)")
        CONFIG_PATH = sys.argv[1]
    benchmark(CONFIG_PATH)
