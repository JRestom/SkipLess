#!/usr/bin/env python3
import os
import math
import time
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F

from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.models import convnext_tiny
from tqdm.auto import tqdm
import yaml

# ============================================================
#  Unpickling helper (your pruning wrapper must exist at import)
# ============================================================
class PrunableConvNeXtBlock(nn.Module):
    """
    Wrapper for ConvNeXt CNBlock used during pruning.
    When disable_main=True, behaves like identity: y = x.
    """
    def __init__(self, block: nn.Module):
        super().__init__()
        self.block = block
        self.disable_main = False

    def forward(self, x):
        if self.disable_main:
            return x
        return self.block(x)

# =====================
#  DDP / utility bits
# =====================
def is_main(rank: int) -> bool:
    return rank == 0

def setup_ddp(rank: int, world_size: int, master_port: str):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = master_port
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    torch.backends.cudnn.benchmark = True
    if is_main(rank):
        print(f"DDP initialized | world_size={world_size}, master_port={master_port}")

def cleanup_ddp():
    if dist.is_initialized():
        dist.destroy_process_group()

def set_seed(seed: int):
    import random, numpy as np
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

# =====================
#  Data / transforms
# =====================
def build_transforms(cfg):
    img_size = cfg["data"].get("img_size", 224)
    val_resize = cfg["data"].get("val_resize", 256)
    mean = cfg["data"].get("mean", [0.485, 0.456, 0.406])
    std = cfg["data"].get("std", [0.229, 0.224, 0.225])

    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(img_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    val_tf = transforms.Compose([
        transforms.Resize(val_resize),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    return train_tf, val_tf

# =====================
#  EMA (optional)
# =====================
class EMA:
    """Exponential Moving Average of model params."""
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.ema = {k: v.detach().clone() for k, v in model.state_dict().items()
                    if v.dtype.is_floating_point}

    @torch.no_grad()
    def update(self, model):
        for k, v in model.state_dict().items():
            if k in self.ema and v.dtype.is_floating_point:
                self.ema[k].mul_(self.decay).add_(v.detach(), alpha=(1.0 - self.decay))

    @torch.no_grad()
    def apply_to(self, model):
        msd = model.state_dict()
        for k, v in self.ema.items():
            if k in msd and msd[k].shape == v.shape:
                msd[k].copy_(v)

# =====================
#  Model helpers
# =====================
def maybe_replace_classifier(model: nn.Module, num_classes: int, rank: int):
    for attr in ["head", "classifier", "fc", "linear"]:
        if hasattr(model, attr):
            head = getattr(model, attr)
            if isinstance(head, nn.Linear) and head.out_features != num_classes:
                new_head = nn.Linear(head.in_features, num_classes)
                nn.init.normal_(new_head.weight, std=0.01)
                nn.init.zeros_(new_head.bias)
                setattr(model, attr, new_head)
                if is_main(rank):
                    print(f"[model] Replaced {attr}: out_features -> {num_classes}")
                return

def build_vanilla_convnext_tiny(num_classes: int):
    m = convnext_tiny(weights=None)
    # torchvision convnext_tiny has .classifier = nn.Sequential(..., nn.Linear(768, num_classes))
    maybe_replace_classifier(m, num_classes, rank=0)
    return m

def _unwrap_state_dict_keys_from_wrapper(sd: dict):
    """
    If weights came from wrapped CNBlocks (PrunableConvNeXtBlock(block=...)),
    keys may contain '.block.'. Normalize to vanilla ConvNeXt by removing it.
    """
    new_sd, changed = {}, False
    for k, v in sd.items():
        if ".block." in k:
            new_sd[k.replace(".block.", ".")] = v
            changed = True
        else:
            new_sd[k] = v
    return new_sd, changed

def load_pruned_artifact(path: str, map_location, cfg, rank: int, device: torch.device) -> nn.Module:
    """
    Accepts:
      - raw nn.Module,
      - dict with {'model': nn.Module, ...}   (your *_full.pt),
      - dict/obj that is a state_dict         (your *_state.pt or similar).
    Returns an nn.Module on device.
    """
    obj = torch.load(path, map_location=map_location)

    # Case 1: already a module
    if isinstance(obj, nn.Module):
        return obj.to(device)

    # Case 2: dict-like
    if isinstance(obj, dict):
        # a) full dict with a model object
        if isinstance(obj.get("model", None), nn.Module):
            return obj["model"].to(device)

        # b) state_dict variants
        state_dict = obj.get("state_dict", None)
        if state_dict is None:
            # directly a state_dict?
            if obj and all(isinstance(v, torch.Tensor) for v in obj.values()):
                state_dict = obj
            else:
                for k in ["model_state_dict", "net_state_dict", "weights"]:
                    if isinstance(obj.get(k), dict):
                        state_dict = obj[k]
                        break
        if state_dict is None:
            raise RuntimeError(f"Couldn't find a state_dict inside keys={list(obj.keys())[:10]}")

        # Normalize keys if they include '.block.'
        state_dict, changed = _unwrap_state_dict_keys_from_wrapper(state_dict)
        if is_main(rank) and changed:
            print("[load] normalized wrapped keys: '.block.' -> '.'")

        num_classes = int(cfg.get("model", {}).get("num_classes", 1000))
        model = build_vanilla_convnext_tiny(num_classes).to(device)

        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if is_main(rank):
            print(f"[load] state_dict loaded with strict=False "
                  f"(missing={len(missing)}, unexpected={len(unexpected)})")
            if missing:    print("  missing (first 10):", missing[:10])
            if unexpected: print("  unexpected(first 10):", unexpected[:10])
        return model

    raise TypeError(f"Unrecognized checkpoint type: {type(obj)} at {path}")

# =====================
#  LR schedule (manual)
# =====================
def lr_at_step(step: int, base_lr: float, warmup_steps: int, total_steps: int) -> float:
    if step < warmup_steps:
        return base_lr * float(step) / float(max(1, warmup_steps))
    # cosine
    progress = (step - warmup_steps) / float(max(1, total_steps - warmup_steps))
    return 0.5 * base_lr * (1.0 + math.cos(math.pi * progress))

def set_optimizer_lr(optimizer: optim.Optimizer, lr: float):
    for g in optimizer.param_groups:
        g["lr"] = lr

# =====================
#  Train / Eval
# =====================
def _unwrap_ddp(model):
    return model.module if isinstance(model, DDP) else model

def train_one_epoch(
    rank, epoch, epochs, model, loader, optimizer, scaler, device, use_amp, ema, criterion,
    base_lr_scaled, warmup_steps, total_steps, start_step
):
    model.train()
    total, correct, loss_sum = 0, 0, 0.0

    pbar = tqdm(total=len(loader), desc=f"Epoch {epoch}/{epochs} [train]", ncols=100, disable=not is_main(rank))

    for batch_idx, (images, targets) in enumerate(loader):
        global_step = start_step + batch_idx
        lr = lr_at_step(global_step, base_lr_scaled, warmup_steps, total_steps)
        set_optimizer_lr(optimizer, lr)

        images = images.to(device, non_blocking=True).to(memory_format=torch.channels_last)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        if use_amp:
            with autocast():
                outputs = model(images)
                loss = criterion(outputs, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

        if ema is not None:
            ema.update(_unwrap_ddp(model))

        loss_sum += loss.item() * targets.size(0)
        correct  += outputs.argmax(1).eq(targets).sum().item()
        total    += targets.size(0)

        if is_main(rank):
            pbar.set_postfix(lr=f"{lr:.2e}", loss=f"{(loss_sum/total):.4f}")
            pbar.update(1)

    if is_main(rank):
        pbar.close()

    return loss_sum / total, 100.0 * correct / total

@torch.no_grad()
def evaluate(rank, model, loader, device, criterion, epoch, epochs):
    model.eval()
    total, correct, loss_sum = 0, 0, 0.0

    pbar = tqdm(loader, desc=f"Epoch {epoch}/{epochs} [val]", ncols=100, disable=not is_main(rank))
    for images, targets in pbar:
        images = images.to(device, non_blocking=True).to(memory_format=torch.channels_last)
        targets = targets.to(device, non_blocking=True)
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss_sum += loss.item() * targets.size(0)
        correct  += outputs.argmax(1).eq(targets).sum().item()
        total    += targets.size(0)
        if is_main(rank):
            pbar.set_postfix(loss=f"{(loss_sum/total):.4f}",
                             acc=f"{(100.0*correct/total):.2f}%")
    if is_main(rank):
        pbar.close()

    return loss_sum / total, 100.0 * correct / total

def save_checkpoints(rank, model, ema, out_dir, run_name, best=False, save_full=True):
    if not is_main(rank):
        return
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    tag = "best" if best else "last"
    to_save = _unwrap_ddp(model)

    # state_dict
    sd_path = Path(out_dir) / f"{run_name}_{tag}.pth"
    torch.save(to_save.state_dict(), str(sd_path))

    # full model (optional)
    if save_full:
        pt_path = Path(out_dir) / f"{run_name}_{tag}.pt"
        torch.save(to_save, str(pt_path))

    # EMA weights (optional)
    if ema is not None:
        ema_sd_path = Path(out_dir) / f"{run_name}_{tag}_ema.pth"
        torch.save(ema.ema, str(ema_sd_path))

    print(f"[Rank {rank}] Saved checkpoints to {out_dir} ({tag})")

# =====================
#  Main worker
# =====================
def main_worker(rank, world_size, args):
    setup_ddp(rank, world_size, args.master_port)

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    # Seed
    set_seed(int(cfg["training"].get("seed", 42)) + rank)

    # Data
    train_tf, val_tf = build_transforms(cfg)
    train_root = cfg["data"]["imagenet_train_path"]
    val_root   = cfg["data"]["imagenet_val_path"]

    train_set = ImageFolder(train_root, transform=train_tf)
    val_set   = ImageFolder(val_root,   transform=val_tf)

    per_gpu_batch = int(cfg["training"]["batch_size"])
    num_workers   = int(cfg["data"].get("num_workers", 8))
    pin_memory    = bool(cfg["data"].get("pin_memory", True))

    train_sampler = DistributedSampler(train_set, num_replicas=world_size, rank=rank, shuffle=True)
    train_loader  = DataLoader(
        train_set,
        batch_size=per_gpu_batch,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        persistent_workers=num_workers > 0,
    )

    if is_main(rank):
        val_loader = DataLoader(
            val_set,
            batch_size=int(cfg["training"].get("val_batch_size", per_gpu_batch)),
            shuffle=False,
            num_workers=max(2, num_workers // 2),
            pin_memory=pin_memory,
            drop_last=False,
            persistent_workers=False,
        )
    else:
        val_loader = None

    device = torch.device(f"cuda:{rank}")

    # Load model (supports *_full.pt dicts or *_state.pt)
    map_location = {f"cuda:{0}": f"cuda:{rank}", "cpu": f"cuda:{rank}"}
    pt_path = cfg["model"]["pretrained_full_pt"]
    model = load_pruned_artifact(pt_path, map_location, cfg, rank, device)

    # Replace classifier if requested num_classes differs
    num_classes = int(cfg["model"].get("num_classes", 1000))
    maybe_replace_classifier(model, num_classes, rank)

    # Channels-last to fix DDP grad-bucket stride warnings & speed up conv kernels
    model = model.to(memory_format=torch.channels_last)

    # Wrap in DDP
    model = DDP(
        model,
        device_ids=[rank],
        find_unused_parameters=bool(cfg["training"].get("find_unused_parameters", False)),
        gradient_as_bucket_view=False,     # avoid grad/bucket stride warnings
        broadcast_buffers=False,
    )

    # Optimizer, LR, AMP, EMA
    base_lr      = float(cfg["training"]["lr"])
    weight_decay = float(cfg["training"].get("weight_decay", 0.05))
    epochs       = int(cfg["training"]["epochs"])
    use_amp      = bool(cfg["training"].get("amp", True))

    global_batch = per_gpu_batch * world_size
    scaled_lr    = base_lr * (global_batch / 256.0)
    if is_main(rank):
        print(f"LR scaling: base_lr={base_lr} -> scaled_lr={scaled_lr} (global_batch={global_batch})")

    optimizer = optim.AdamW(model.parameters(), lr=scaled_lr, weight_decay=weight_decay)
    scaler    = GradScaler(enabled=use_amp)

    ema_cfg = cfg.get("ema", {"enable": False})
    ema     = EMA(_unwrap_ddp(model), decay=float(ema_cfg.get("decay", 0.999))) if ema_cfg.get("enable", False) else None

    label_smoothing = float(cfg["training"].get("label_smoothing", 0.0))
    criterion       = nn.CrossEntropyLoss(label_smoothing=label_smoothing).to(device)

    # W&B (rank 0 only)
    wb = cfg.get("wandb", {"enable": False})
    if is_main(rank) and wb.get("enable", False):
        import wandb
        wandb.init(
            project=wb.get("project", "imagenet-ft"),
            entity=wb.get("entity", None),
            name=wb.get("run_name", f"ft_convnext_{int(time.time())}"),
            config=cfg,
        )

    # Output
    out_dir  = cfg["training"].get("output_dir", "./outputs")
    run_name = cfg.get("run_name", "convnext_ft")
    save_full = bool(cfg["training"].get("save_full_model", True))

    # Manual LR schedule params
    total_steps  = len(train_loader) * epochs
    warmup_ratio = float(cfg["sched"].get("warmup_ratio", 0.1))
    warmup_steps = max(1, int(warmup_ratio * total_steps))

    # Train
    best_acc = -1.0
    for epoch in range(1, epochs + 1):
        train_sampler.set_epoch(epoch)
        start_step = (epoch - 1) * len(train_loader)

        tr_loss, tr_acc = train_one_epoch(
            rank=rank,
            epoch=epoch,
            epochs=epochs,
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            use_amp=use_amp,
            ema=ema,
            criterion=criterion,
            base_lr_scaled=scaled_lr,
            warmup_steps=warmup_steps,
            total_steps=total_steps,
            start_step=start_step,
        )

        if is_main(rank):
            log = {"epoch": epoch, "train/loss": tr_loss, "train/acc": tr_acc, "lr": optimizer.param_groups[0]["lr"]}
            if wb.get("enable", False):
                import wandb
                wandb.log(log)
            else:
                print(log)

            # Evaluate (EMA-applied if enabled)
            if ema is not None:
                ema.apply_to(_unwrap_ddp(model))
            val_loss, val_acc = evaluate(rank, model, val_loader, device, criterion, epoch, epochs)
            if wb.get("enable", False):
                import wandb
                wandb.log({"val/loss": val_loss, "val/acc": val_acc})
            else:
                print({"val/loss": val_loss, "val/acc": val_acc})

            # Save last & best
            save_checkpoints(rank, model, ema, out_dir, run_name, best=False, save_full=save_full)
            if val_acc > best_acc:
                best_acc = val_acc
                save_checkpoints(rank, model, ema, out_dir, run_name, best=True, save_full=save_full)

    if is_main(rank) and wb.get("enable", False):
        import wandb
        wandb.finish()

    cleanup_ddp()

# =====================
#  Entrypoint
# =====================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--master_port", type=str, default=os.environ.get("MASTER_PORT", "29500"))
    args = parser.parse_args()

    # Heads-up about NumPy 2 if present
    try:
        import numpy as _np
        if tuple(map(int, _np.__version__.split(".")[:2])) >= (2, 0):
            if os.environ.get("NUMPY_OK_WITH_TORCH", "") != "1":
                print("Note: NumPy >= 2 detected. If you see compatibility issues, install numpy<2.")
    except Exception:
        pass

    world_size = torch.cuda.device_count()
    assert world_size >= 1, "No CUDA device found."
    mp.spawn(main_worker, args=(world_size, args), nprocs=world_size, join=True)

if __name__ == "__main__":
    main()
