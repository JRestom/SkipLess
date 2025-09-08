#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ConvNeXt Top-K Block Pruning (with optional ablation reuse)

- Wraps torchvision ConvNeXt CNBlocks with a prunable adapter.
- Scores each block by accuracy drop when its main transform is disabled (Bi(x)=x).
- Prunes the K least-important blocks by replacing them with nn.Identity.
- Saliency is computed on a *deterministic* subset of the TRAIN split (no augmentation) unless
  you provide an existing ablation JSON via `ablation_json_in` in the YAML.
- Can emit *multiple K variants in one run* using `generate_many_k: [1,2,3,...]`.

Saves per K: state_dict, full (pickled model), manifest.json (incl. #params).
Optionally saves ablation JSON when computed.

Author: you + ChatGPT
"""

import os
import json
import yaml
import argparse
import random
from collections import defaultdict
import copy

import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader, Subset
from torchvision import transforms, datasets
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights
from tqdm import tqdm


# ---------------------- Reproducibility ----------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True


# ---------------------- Module utilities ----------------------
def set_module_by_name(model: nn.Module, name: str, new_module: nn.Module):
    parts = name.split('.')
    parent = model
    for p in parts[:-1]:
        parent = getattr(parent, p)
    setattr(parent, parts[-1], new_module)


class PrunableConvNeXtBlock(nn.Module):
    """
    Wraps a torchvision ConvNeXt CNBlock so that when disable_main=True
    the block behaves as Bi(x) = x (i.e., Fi(x) = 0), preserving identity.
    """
    def __init__(self, block: nn.Module):
        super().__init__()
        self.block = block
        self.disable_main = False

    def forward(self, x):
        if self.disable_main:
            return x  # correct ablation: keep the residual, drop the transform
        return self.block(x)


def wrap_blocks(model: nn.Module):
    """Collect CNBlock names first, then replace to avoid mutating during iteration."""
    to_wrap = [name for name, m in model.named_modules() if type(m).__name__ == "CNBlock"]
    module_map = dict(model.named_modules())
    for name in to_wrap:
        orig = module_map[name]
        set_module_by_name(model, name, PrunableConvNeXtBlock(orig))
        print(f"✅ Wrapped CNBlock as prunable: {name}")


def get_prunable_blocks(model: nn.Module):
    return [name for name, m in model.named_modules() if hasattr(m, "disable_main")]


# ---------------------- Data ----------------------
def _val_style_transform():
    # Deterministic, no augmentation — val-style preprocessing
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])


def get_train_subset_loader(
    data_root: str,
    batch_size: int,
    samples_per_class: int = None,
    ratio: float = None,
    num_workers: int = 8,
    seed: int = 42,
    pin_memory: bool = True,
):
    """
    Deterministic scoring subset drawn from the TRAIN split (as requested),
    but using val-style transforms (no augmentation) for stable saliency.

    Specify either:
      - samples_per_class (preferred), or
      - ratio (global fraction of train set).
    """
    transform = _val_style_transform()
    train_dir = os.path.join(data_root, "train")
    dataset = datasets.ImageFolder(train_dir, transform=transform)

    rng = np.random.RandomState(seed)
    if samples_per_class is not None:
        cls_to_indices = defaultdict(list)
        for idx, (_, label) in enumerate(dataset.samples):
            cls_to_indices[label].append(idx)
        selected = []
        for label, idxs in cls_to_indices.items():
            idxs = np.array(idxs)
            rng.shuffle(idxs)
            k = min(samples_per_class, len(idxs))
            selected.extend(idxs[:k].tolist())
    elif ratio is not None:
        all_idx = np.arange(len(dataset))
        rng.shuffle(all_idx)
        k = max(1, int(len(dataset) * float(ratio)))
        selected = all_idx[:k].tolist()
    else:
        raise ValueError("Specify either `samples_per_class` or `ratio` for the train subset.")

    subset = Subset(dataset, selected)
    return DataLoader(subset, batch_size=batch_size, shuffle=False,
                      num_workers=num_workers, pin_memory=pin_memory)


def get_val_loader(
    data_root: str,
    batch_size: int,
    num_workers: int = 8,
    pin_memory: bool = True,
):
    transform = _val_style_transform()
    val_dir = os.path.join(data_root, "val")
    dataset = datasets.ImageFolder(val_dir, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False,
                      num_workers=num_workers, pin_memory=pin_memory)


# ---------------------- Eval ----------------------
@torch.inference_mode()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct, total = 0, 0
    for imgs, labels in loader:
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(imgs)
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)
    return 100.0 * correct / total if total > 0 else 0.0


# ---------------------- Saliency & Pruning ----------------------
def ablation_study(model: nn.Module, loader: DataLoader, device: torch.device, baseline_acc: float):
    module_map = dict(model.named_modules())
    blocks = [n for n, m in module_map.items() if hasattr(m, "disable_main")]
    results = {}
    for name in tqdm(blocks, desc="Ablating blocks", dynamic_ncols=True):
        blk = module_map[name]
        blk.disable_main = True
        drop = baseline_acc - evaluate(model, loader, device)
        results[name] = float(drop)
        blk.disable_main = False
    return results


def finalize_pruned_blocks(model: nn.Module):
    module_map = dict(model.named_modules())
    for name, m in module_map.items():
        if isinstance(m, PrunableConvNeXtBlock) and m.disable_main:
            set_module_by_name(model, name, nn.Identity())
            print(f"🔁 Finalized prune → Identity: {name}")


# ---------------------- Param counting ----------------------
def count_parameters(model: nn.Module):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


# ---------------------- Main ----------------------
def main():
    parser = argparse.ArgumentParser(description="ConvNeXt Top-K Block Pruning (with ablation reuse)")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    seed = int(cfg.get("seed", 42))
    set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cudnn.benchmark = True  # OK for inference/eval loops

    # Model
    model = convnext_tiny(weights=ConvNeXt_Tiny_Weights.IMAGENET1K_V1).to(device)
    wrap_blocks(model)

    print("\n✅ Prunable CNBlocks found:")
    for name in get_prunable_blocks(model):
        print(" •", name)

    # Optional checkpoint (fine-tuned weights)
    if cfg.get("checkpoint_path"):
        print(f"\n📦 Loading checkpoint: {cfg['checkpoint_path']}")
        state = torch.load(cfg["checkpoint_path"], map_location="cpu")
        if isinstance(state, dict) and "state_dict" in state:
            model.load_state_dict(state["state_dict"], strict=True)
        elif isinstance(state, dict):
            model.load_state_dict(state, strict=True)
        else:
            model = state.to(device)

    # Data loaders (only needed if we compute ablation here)
    data_root = cfg["data_root"]
    batch_size = int(cfg.get("batch_size", 128))
    num_workers = int(cfg.get("num_workers", 8))
    pin_memory = bool(cfg.get("pin_memory", True))

    # Load or compute ablation scores
    results = None
    if cfg.get("ablation_json_in"):
        with open(cfg["ablation_json_in"], "r") as f:
            results = json.load(f)
        print(f"\n📥 Loaded ablation scores from: {cfg['ablation_json_in']}")
    else:
        print("\n🧮 Computing ablation scores on deterministic TRAIN subset...")
        train_subset_loader = get_train_subset_loader(
            data_root=data_root,
            batch_size=batch_size,
            samples_per_class=cfg.get("subset_samples_per_class"),
            ratio=cfg.get("subset_ratio"),
            num_workers=num_workers,
            seed=seed,
            pin_memory=pin_memory,
        )
        baseline_acc = evaluate(model, train_subset_loader, device)
        print(f"✅ Baseline accuracy on train-subset (deterministic): {baseline_acc:.2f}%")
        results = ablation_study(model, train_subset_loader, device, baseline_acc)
        if cfg.get("save_ablation_json"):
            os.makedirs(os.path.dirname(cfg["save_ablation_json"]), exist_ok=True)
            with open(cfg["save_ablation_json"], "w") as f:
                json.dump(results, f, indent=2)
            print(f"💾 Saved ablation scores to {cfg['save_ablation_json']}")

    # Sort once (ascending drop)
    sorted_drops = sorted(results.items(), key=lambda x: x[1])

    # Decide which K values to build
    Ks = cfg.get("generate_many_k")
    if Ks is None:
        Ks = [int(cfg.get("k_blocks_to_prune", 5))]
    Ks = list(map(int, Ks))

    # Prepare loaders for final evaluation (val)
    val_loader = get_val_loader(
        data_root=data_root,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    # Keep a clean baseline state to rebuild each K independently
    base_state = copy.deepcopy(model.state_dict())

    # Output dir and model name
    out_dir = cfg.get("output_dir", ".")
    os.makedirs(out_dir, exist_ok=True)
    model_name = cfg.get("model_name", "convnext_tiny")

    for K in Ks:
        print(f"\n=== Building pruned model for K={K} blocks ===")
        # Restore baseline and re-wrap (wrappers may be replaced by Identities each loop)
        model.load_state_dict(base_state)
        wrap_blocks(model)  # ensure prunable wrappers are present

        # Mark first K blocks to prune and finalize
        module_map = dict(model.named_modules())
        to_prune = sorted_drops[:K]
        print("✂️  Pruning plan:")
        for name, drop in to_prune:
            print(f" • {name}  (drop: {drop:.4f}%)")
            module_map[name].disable_main = True
        finalize_pruned_blocks(model)

        # Evaluate on VAL
        pruned_val_acc = evaluate(model, val_loader, device)

        # Count params after pruning
        n_params, n_params_trainable = count_parameters(model)

        print(f"✅ K={K}: VAL acc={pruned_val_acc:.2f}% | Total params={n_params:,} | Trainable={n_params_trainable:,}")

        # Save artifacts
        save_state = os.path.join(out_dir, f"{model_name}_pruned_{K}blocks_state.pt")
        save_full  = os.path.join(out_dir, f"{model_name}_pruned_{K}blocks_full.pt")
        manifest_path = os.path.join(out_dir, f"{model_name}_pruned_{K}blocks_manifest.json")

        torch.save(model.state_dict(), save_state)
        torch.save({"model": model, "cfg": cfg, "pruned_blocks": [n for n, _ in to_prune]}, save_full)

        manifest = {
            "model_name": model_name,
            "k_pruned": K,
            "pruned_blocks": [n for n, _ in to_prune],
            "pruned_val_acc": pruned_val_acc,
            "seed": seed,
            "n_params": int(n_params),
            "n_params_trainable": int(n_params_trainable),
        }
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)

        print(f"💾 Saved K={K}:")
        print(f"  • state_dict: {save_state}")
        print(f"  • full:       {save_full}")
        print(f"  • manifest:   {manifest_path}")


if __name__ == "__main__":
    main()
