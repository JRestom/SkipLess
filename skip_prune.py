import os
import sys
import json
import yaml
import random
import argparse
from types import MethodType
from collections import defaultdict

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import CIFAR10, ImageFolder
from tqdm import tqdm
from fvcore.nn import FlopCountAnalysis

# ----- Your models import -----
from models import resnet18, resnet34, resnet50, resnet101, resnet152


# ================================
# Util: seeds, FLOPs, params, eval
# ================================
def set_seed(seed: int):
    if seed is None:
        return
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def print_model_flops_and_params(model, input_shape=(1, 3, 224, 224)):
    model.eval()
    device = next(model.parameters()).device
    dummy_input = torch.randn(input_shape, device=device)
    flops = FlopCountAnalysis(model, dummy_input)
    print("🧮 FLOPs and parameter count:")
    try:
        total = flops.total()
        print(f"Total FLOPs: {total / 1e9:.2f} GFLOPs")
    except Exception as e:
        print(f"FLOP analysis failed: {e}")


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        correct += outputs.argmax(1).eq(targets).sum().item()
        total += targets.size(0)
    return 100.0 * correct / total if total > 0 else 0.0


# =========================
# Dataset / dataloaders
# =========================
def get_dataloaders(cfg):
    dataset = cfg["dataset"].lower()
    data_root = cfg["data_root"]
    bs = cfg["batch_size"]

    if dataset == "cifar10":
        t_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.247, 0.243, 0.261))
        ])
        t_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.247, 0.243, 0.261))
        ])
        train = CIFAR10(data_root, train=True, download=True, transform=t_train)
        val = CIFAR10(data_root, train=False, download=True, transform=t_test)

    elif dataset == "imagenet":
        # Use this branch for ImageNet OR any ImageFolder dataset like CUB
        t_train = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])
        t_test = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])
        train = ImageFolder(os.path.join(data_root, "train"), transform=t_train)
        val = ImageFolder(os.path.join(data_root, "val"), transform=t_test)

        # Optional sub-sample of TRAIN for speed (used in older flow)
        if cfg.get("val_subset_ratio", 1.0) < 1.0:
            ratio = float(cfg["val_subset_ratio"])
            indices = list(range(len(train)))
            random.shuffle(indices)
            train = Subset(train, indices[:int(len(indices) * ratio)])

    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

    # NOTE: We always evaluate/score on the VAL loader for reliability.
    train_loader = DataLoader(train, bs, shuffle=False, num_workers=16, pin_memory=True)
    val_loader = DataLoader(val, bs, shuffle=False, num_workers=16, pin_memory=True)
    return train_loader, val_loader


# =========================
# ResNet block detection
# =========================
def is_resnet_block(m: nn.Module) -> bool:
    # torchvision-style BasicBlock/Bottleneck both have conv1 & conv2
    return hasattr(m, "conv1") and hasattr(m, "conv2")


def _get_stride(s):
    return s[0] if isinstance(s, tuple) else s


def is_identity_safe_resnet_block(block: nn.Module) -> bool:
    """
    Identity replacement is safe iff:
      - stride == 1
      - in_channels == out_channels
      - no downsample/projection path
    """
    # Bottleneck: conv1, conv2, conv3 exist
    if hasattr(block, "conv1") and hasattr(block, "conv3"):
        in_c = block.conv1.in_channels
        out_c = block.conv3.out_channels
        s1 = _get_stride(block.conv1.stride)
        s2 = _get_stride(block.conv2.stride)
        stride = max(s1, s2)
        has_down = hasattr(block, "downsample") and block.downsample is not None
        return (stride == 1) and (in_c == out_c) and (not has_down)

    # BasicBlock: conv1, conv2, no conv3
    if hasattr(block, "conv1") and hasattr(block, "conv2") and not hasattr(block, "conv3"):
        in_c = block.conv1.in_channels
        out_c = block.conv2.out_channels
        s1 = _get_stride(block.conv1.stride)
        s2 = _get_stride(block.conv2.stride)
        stride = max(s1, s2)
        has_down = hasattr(block, "downsample") and block.downsample is not None
        return (stride == 1) and (in_c == out_c) and (not has_down)

    return False


def _module_device_dtype(mod: nn.Module):
    for p in mod.parameters(recurse=True):
        return p.device, p.dtype
    return torch.device("cpu"), torch.float32


# =========================
# Block replacement
# =========================
class Residual1x1(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, kernel_size=1, bias=False)
        self.match = nn.Identity() if in_c == out_c else nn.Conv2d(in_c, out_c, 1, bias=False)

    def forward(self, x):
        return self.conv(x) + self.match(x)


def get_block_io_channels(block):
    if hasattr(block, "conv1") and hasattr(block, "conv3"):
        return block.conv1.in_channels, block.conv3.out_channels
    elif hasattr(block, "conv1") and hasattr(block, "conv2"):
        return block.conv1.in_channels, block.conv2.out_channels
    else:
        raise ValueError("Unrecognized block type")


def replace_block(model, name, type="identity", in_c=None, out_c=None):
    parts = name.split(".")
    parent = model
    for p in parts[:-1]:
        parent = getattr(parent, p)
    old = getattr(parent, parts[-1])

    if type == "identity":
        # Safer path: only allow if identity-safe
        if not is_identity_safe_resnet_block(old):
            raise ValueError(f"{name} is NOT identity-safe (stride/channel/downsample).")
        dev, dt = _module_device_dtype(old)
        new_block = nn.Identity().to(device=dev, dtype=dt)
    elif type == "conv1x1":
        new_block = nn.Conv2d(in_c, out_c, kernel_size=1, bias=False)
        new_block.to(*_module_device_dtype(old))
    elif type == "residual1x1":
        new_block = Residual1x1(in_c, out_c)
        new_block.to(*_module_device_dtype(old))
    else:
        raise ValueError(f"Unknown replacement type: {type}")

    setattr(parent, parts[-1], new_block)


# =========================
# Ablation via forward patch
# =========================
def ablation_study(model, loader, device, baseline_acc):
    """
    For each ResNet block:
      - Temporarily patch forward to return only the shortcut path
      - Measure accuracy drop vs baseline
      - Restore original forward
    """
    # Collect candidate blocks
    blocks = [(name, module) for name, module in model.named_modules() if is_resnet_block(module)]
    if len(blocks) == 0:
        print("⚠️ No ResNet-like blocks found to ablate.")
        return {}

    results = {}

    for name, block in tqdm(blocks, desc="Ablating blocks"):
        orig_forward = block.forward

        def skip_main_forward(self, x):
            if hasattr(self, "downsample") and self.downsample is not None:
                identity = self.downsample(x)
            else:
                identity = x
            return identity  # no main path, no post-ReLU

        # Monkey-patch
        block.forward = MethodType(skip_main_forward, block)

        # Evaluate with this block ablated
        drop = baseline_acc - evaluate(model, loader, device)
        results[name] = drop

        # Restore
        block.forward = orig_forward

    return results


# =========================
# Delta utilities (NEW)
# =========================
def _stage_of(block_name: str) -> str:
    # 'layer3.4' -> 'layer3'
    return block_name.split(".")[0] if "." in block_name else "root"

def _model_order_block_list(model) -> list:
    """Return blocks in true model order (named_modules traversal)."""
    return [name for name, m in model.named_modules() if is_resnet_block(m)]

def _compute_block_deltas(block_order: list, base_scores: dict, window: int = 1) -> dict:
    """
    First-difference over the ordered block list:
        d[i] = base[i] - base[i-1]
    Keep only positive deltas (protective). Optionally average over ±window neighbors.
    Returns dict block_name -> positive_delta_mean (>=0).
    """
    n = len(block_order)
    if n == 0:
        return {}

    # sequence of base scores in order
    seq = [base_scores.get(b, None) for b in block_order]
    # robust fill: if any missing, set to neighbor value
    for i in range(n):
        if seq[i] is None:
            seq[i] = seq[i-1] if i > 0 else 0.0

    # raw deltas (first element 0)
    deltas = [0.0]
    for i in range(1, n):
        deltas.append(seq[i] - seq[i-1])

    # positive deltas with local averaging
    pos_mean = {}
    for i, b in enumerate(block_order):
        lo = max(0, i - window)
        hi = min(n - 1, i + window)
        local = deltas[lo:hi + 1]
        positives = [x for x in local if x > 0]
        pos_mean[b] = sum(positives) / len(positives) if len(positives) > 0 else 0.0
    return pos_mean

def _normalize(values: dict, mode: str, stage_of_map: dict) -> dict:
    """
    Normalize values in-place by: none | global | stage.
    """
    if mode == "none":
        return dict(values)

    if mode == "global":
        xs = list(values.values())
        mn, mx = min(xs), max(xs)
        if mx - mn < 1e-12:
            return {k: 0.0 for k in values}
        return {k: (v - mn) / (mx - mn) for k, v in values.items()}

    if mode == "stage":
        out = {}
        buckets = defaultdict(list)
        for b, v in values.items():
            buckets[stage_of_map[b]].append(v)
        for b, v in values.items():
            arr = buckets[stage_of_map[b]]
            mn, mx = min(arr), max(arr)
            if mx - mn < 1e-12:
                out[b] = 0.0
            else:
                out[b] = (v - mn) / (mx - mn)
        return out

    return dict(values)

def delta_adjust_scores(model, base_scores: dict, lam: float, window: int, normalize: str) -> dict:
    """
    Adjust base (ablation-drop) scores with positive delta penalties.
      adjusted[b] = base[b] + lam * penalty[b]
    """
    order = _model_order_block_list(model)
    penalty = _compute_block_deltas(order, base_scores, window=window)
    stage_of_map = {b: _stage_of(b) for b in order}
    penalty = _normalize(penalty, normalize, stage_of_map)
    adjusted = {}
    for b in base_scores.keys():
        adjusted[b] = base_scores[b] + lam * penalty.get(b, 0.0)
    return adjusted


# =========================
# Main
# =========================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg.get("seed", 42))
    cudnn.benchmark = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model
    model_map = {
        "resnet18": resnet18, "resnet34": resnet34, "resnet50": resnet50,
        "resnet101": resnet101, "resnet152": resnet152
    }
    model_name = cfg["model_name"]
    if model_name not in model_map:
        raise ValueError(f"Unknown model_name '{model_name}'. Allowed: {list(model_map.keys())}")

    model = model_map[model_name](num_classes=cfg.get("num_classes", 1000))

    # Load checkpoint (state_dict preferred; support full model as fallback)
    ckpt_path = cfg["checkpoint"]
    try:
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        if isinstance(checkpoint, dict) and any(
            k.startswith("module.") or k.startswith("conv1.weight") or k.endswith(".weight")
            for k in checkpoint.keys()
        ):
            model.load_state_dict(checkpoint, strict=True)
        elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            model.load_state_dict(checkpoint["state_dict"], strict=False)
        else:
            # Maybe it's a full serialized model
            print("ℹ️ Loading full serialized model object from checkpoint.")
            model = checkpoint
    except RuntimeError as e:
        print(f"❌ Failed to load model weights: {e}")
        sys.exit(1)

    model.to(device)

    # Data
    train_loader, val_loader = get_dataloaders(cfg)

    # Baseline on VAL (more reliable importance)
    baseline_acc = evaluate(model, val_loader, device)
    print(f"Baseline accuracy (val): {baseline_acc:.2f}%")

    # Ablation on VAL -> base per-block scores (drop)
    results = ablation_study(model, val_loader, device, baseline_acc)
    print(f"\nScored {len(results)} blocks.")

    # Report least harmful by base score
    if len(results) > 0:
        print("\nLeast harmful blocks (smallest accuracy drop, base method):")
        for block, drop in sorted(results.items(), key=lambda x: x[1])[:min(10, len(results))]:
            print(f"{block}: drop {drop:.2f}%")
    else:
        print("⚠️ No scored blocks; pruning will be skipped.")

    # FLOPs & params BEFORE
    print("\n📊 Model stats before pruning:")
    input_shape = (1, 3, 32, 32) if cfg["dataset"].lower() == "cifar10" else (1, 3, 224, 224)
    print_model_flops_and_params(model, input_shape)
    print(f"Params before: {count_parameters(model)/1e6:.2f}M")

    # =========================
    # Delta-aware ranking (NEW)
    # =========================
    delta_cfg = cfg.get("delta", {})
    use_delta = bool(delta_cfg.get("use", False))
    lam = float(delta_cfg.get("lambda", 0.5))
    win = int(delta_cfg.get("window", 1))
    norm_mode = str(delta_cfg.get("normalize", "stage")).lower()  # none|global|stage

    # Base scores = ablation drops (lower is better to prune)
    base_scores = dict(results)

    if use_delta and len(base_scores) > 0:
        print(f"\n🔧 Delta mode ON  (lambda={lam}, window={win}, normalize={norm_mode})")
        final_scores = delta_adjust_scores(model, base_scores, lam=lam, window=win, normalize=norm_mode)

        # Show top-10 least important by adjusted score
        print("\nLeast harmful blocks (delta-adjusted):")
        preview = sorted(final_scores.items(), key=lambda x: x[1])[:min(10, len(final_scores))]
        for b, adj in preview:
            base = base_scores.get(b, float('nan'))
            print(f"{b}: adj {adj:.3f}, base {base:.3f}")
    else:
        if use_delta:
            print("\n⚠️ Delta requested but no base scores available; falling back to base.")
        final_scores = base_scores

    # Choose K blocks to prune
    k = int(cfg["k_blocks_to_prune"])
    replacement_type = cfg.get("replacement_type", "identity")

    # Optional heuristic to avoid typical downsample blocks named '*.0'
    # (We already guard identity safety, but keep this heuristic too.)
    candidates = [(n, final_scores[n]) for n in final_scores.keys() if not n.endswith(".0")]
    candidates = sorted(candidates, key=lambda x: x[1])  # ascending score = least harmful first

    pruned = 0
    for block_name, score in candidates:
        if pruned >= k:
            break
        block = dict(model.named_modules())[block_name]
        try:
            if replacement_type == "identity":
                if not is_identity_safe_resnet_block(block):
                    print(f"⏭️  Skipping {block_name}: not identity-safe.")
                    continue
                print(f"Pruning {block_name} (score {score:.2f}) -> Identity")
                replace_block(model, block_name, type="identity")
            else:
                in_c, out_c = get_block_io_channels(block)
                print(f"Pruning {block_name} (score {score:.2f}) -> {replacement_type}")
                replace_block(model, block_name, type=replacement_type, in_c=in_c, out_c=out_c)
            pruned += 1
        except Exception as e:
            print(f"⚠️ Could not prune {block_name}: {e}")

    print(f"\n✅ Pruned {pruned} blocks (requested {k}).")

    # Eval AFTER pruning
    acc = evaluate(model, val_loader, device)
    print(f"Accuracy after pruning (val): {acc:.2f}%")

    # FLOPs & params AFTER
    print("\n📊 Model stats after pruning:")
    print_model_flops_and_params(model, input_shape)
    print(f"Params after: {count_parameters(model)/1e6:.2f}M")

    # Save whole model object (keeps structure after replacements)
    out_name = f"{cfg['model_name']}_pruned_top_{pruned}_{cfg['dataset'].lower()}.pt"
    torch.save(model, out_name)
    print(f"✅ Pruned model saved to {out_name}")


if __name__ == "__main__":
    main()
