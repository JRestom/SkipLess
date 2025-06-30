import os
import sys
import json
import torch
import argparse
import random
import yaml
from collections import defaultdict
from tqdm import tqdm
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Subset
import torch.nn as nn
import subprocess

from models import resnet18, resnet34, resnet50, resnet101, resnet152

# ====================== Utilities ======================

def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            correct += outputs.argmax(1).eq(targets).sum().item()
            total += targets.size(0)
    return 100. * correct / total

def ablation_study(model, loader, device, baseline_acc):
    blocks = [name for name, block in model.named_modules() if hasattr(block, 'disable_main')]
    results = {}
    for name in tqdm(blocks, desc="Ablating blocks"):
        block = dict(model.named_modules())[name]
        block.disable_main = True
        acc_drop = baseline_acc - evaluate(model, loader, device)
        results[name] = acc_drop
        block.disable_main = False
    return results

def get_block_io_channels(block):
    if hasattr(block, 'conv1') and hasattr(block, 'conv3'):
        return block.conv1.in_channels, block.conv3.out_channels
    elif hasattr(block, 'conv1') and hasattr(block, 'conv2'):
        return block.conv1.in_channels, block.conv2.out_channels
    else:
        raise ValueError("Unrecognized block type")

def replace_block(model, name, type="identity", in_c=None, out_c=None):
    parts = name.split('.')
    mod = model
    for p in parts[:-1]:
        mod = getattr(mod, p)
    if type == "identity":
        new_block = nn.Identity()
    elif type == "conv1x1":
        new_block = nn.Conv2d(in_c, out_c, kernel_size=1)
    elif type == "residual1x1":
        new_block = Residual1x1(in_c, out_c)
    else:
        raise ValueError(f"Unknown replacement type: {type}")
    setattr(mod, parts[-1], new_block)

def get_imagenet_dataloaders(cfg):
    t_train = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    t_val = transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train = ImageFolder(os.path.join(cfg['data_root'], "train"), transform=t_train)
    val = ImageFolder(os.path.join(cfg['data_root'], "val"), transform=t_val)

    subset_ratio = cfg.get('subset_ratio', 0.1)
    if subset_ratio < 1.0:
        indices = list(range(len(train)))
        random.shuffle(indices)
        subset_size = int(len(indices) * subset_ratio)
        train = Subset(train, indices[:subset_size])

    train_loader = DataLoader(train, batch_size=cfg['batch_size'], shuffle=False, num_workers=16, pin_memory=True)
    val_loader = DataLoader(val, batch_size=cfg['batch_size'], shuffle=False, num_workers=16, pin_memory=True)
    return train_loader, val_loader

# ====================== Main ======================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)

    mode = cfg['mode']
    assert mode in ['iterative', 'iterative_finetune']

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_map = {
        'resnet18': resnet18, 'resnet34': resnet34, 'resnet50': resnet50,
        'resnet101': resnet101, 'resnet152': resnet152
    }
    model = model_map[cfg['model_name']](num_classes=cfg['num_classes'])
    model.load_state_dict(torch.load(cfg['checkpoint_path']))
    model.to(device)

    train_loader, val_loader = get_imagenet_dataloaders(cfg)
    baseline_acc = evaluate(model, train_loader, device)
    print(f"📊 Baseline accuracy: {baseline_acc:.2f}%")

    k = cfg['k_blocks_to_prune']
    replacement_type = cfg['replacement_type']
    finetune_epochs = cfg.get('finetune_epochs', 5)

    all_ablation_scores = []

    for step in range(k):
        print(f"\n🔁 Iteration {step + 1}/{k}")
        ablation = ablation_study(model, train_loader, device, baseline_acc)
        candidates = [(n, d) for n, d in ablation.items() if not n.endswith('.0')]
        block_name, drop = sorted(candidates, key=lambda x: x[1])[0]

        all_ablation_scores.append({
            f"step_{step}": dict(sorted(ablation.items(), key=lambda x: x[1], reverse=True))
        })

        block = dict(model.named_modules())[block_name]
        in_c, out_c = get_block_io_channels(block)
        replace_block(model, block_name, replacement_type, in_c, out_c)
        print(f"🧹 Pruned block {block_name} (drop {drop:.2f}%)")

        if mode == 'iterative_finetune':
            temp_path = f"temp_model_step{step}.pt"
            torch.save(model, temp_path)

            # ──  load finetune YAML, inject path ───────────────
            with open(cfg['finetune_config'], 'r') as f:
                ft_cfg = yaml.safe_load(f)

            ft_cfg['training']['pretrained_weights'] = temp_path

            #  write it back
            with open(cfg["finetune_config"], "w") as f:
                yaml.safe_dump(ft_cfg, f)

            # ── launch distributed training script ─────────────
            subprocess.run([
                "python", "distributed_training.py",
                "--config", cfg['finetune_config']
            ])

            # ──  reload the updated weights ─────────────────────
            model = torch.load(temp_path).to(device)

    acc = evaluate(model, val_loader, device)
    print(f"\n✅ Final accuracy after pruning: {acc:.2f}%")

    out_path = cfg.get('save_path', f"{cfg['model_name']}_{mode}.pt")
    torch.save(model, out_path)
    print(f"💾 Final model saved to {out_path}")

    ranking_path = cfg.get("ranking_json", "ranking_scores_iterative.json")
    with open(ranking_path, 'w') as f:
        json.dump(all_ablation_scores, f, indent=2)
    print(f"📄 Final block ranking scores saved to: {ranking_path}")

if __name__ == "__main__":
    main()
