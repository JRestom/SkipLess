import os
import sys
import json
import torch
import random
import argparse
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from tqdm import tqdm
from torchvision import transforms
from torchvision.datasets import CIFAR10, ImageFolder
from torch.utils.data import DataLoader, Subset
from fvcore.nn import FlopCountAnalysis
import yaml

from models import resnet18, resnet34, resnet50, resnet101, resnet152, resnet56


def print_model_flops_and_params(model, input_shape=(1, 3, 224, 224)):
    model.eval()
    dummy_input = torch.randn(input_shape).to(next(model.parameters()).device)
    flops = FlopCountAnalysis(model, dummy_input)
    print("🧮 FLOPs and parameter count:")
    print(f"Total FLOPs: {flops.total() / 1e9:.2f} GFLOPs")

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

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

def get_block_io_channels(block):
    if hasattr(block, 'conv1') and hasattr(block, 'conv3'):
        return block.conv1.in_channels, block.conv3.out_channels
    elif hasattr(block, 'conv1') and hasattr(block, 'conv2'):
        return block.conv1.in_channels, block.conv2.out_channels
    else:
        raise ValueError("Unrecognized block type")

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

class Residual1x1(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, kernel_size=1)
        self.match = nn.Identity() if in_c == out_c else nn.Conv2d(in_c, out_c, 1)
    def forward(self, x):
        return self.conv(x) + self.match(x)

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

def get_dataloaders(cfg):
    if cfg['dataset'] == "cifar10":
        t_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
        ])
        t_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
        ])
        train = CIFAR10(cfg['data_root'], train=True, download=True, transform=t_train)
        val = CIFAR10(cfg['data_root'], train=False, download=True, transform=t_test)

    elif cfg['dataset'] == "imagenet":
        t_train = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        t_test = transforms.Compose([
            transforms.Resize(256), transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        train = ImageFolder(os.path.join(cfg['data_root'], "train"), transform=t_train)
        val = ImageFolder(os.path.join(cfg['data_root'], "val"), transform=t_test)

        if cfg.get('val_subset_ratio', 1.0) < 1.0:
            indices = list(range(len(train)))
            random.shuffle(indices)
            train = Subset(train, indices[:int(len(indices) * cfg['val_subset_ratio'])])

    else:
        raise ValueError("Unsupported dataset")

    return DataLoader(train, cfg['batch_size'], shuffle=False, num_workers=16, pin_memory=True), \
           DataLoader(val, cfg['batch_size'], shuffle=False, num_workers=16, pin_memory=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)

    cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model
    model_map = {
        'resnet18': resnet18, 'resnet34': resnet34, 'resnet50': resnet50,
        'resnet101': resnet101, 'resnet152': resnet152, 'resnet56': resnet56
    }
    model = model_map[cfg['model_name']](num_classes=cfg.get('num_classes', 1000))

    try:
        model.load_state_dict(torch.load(cfg['checkpoint']), strict=True)

    except RuntimeError as e:
        print(f"❌ Failed to load model weights: {e}")
        sys.exit(1)

    model.to(device)

    train_loader, val_loader = get_dataloaders(cfg)

    baseline_acc = evaluate(model, train_loader, device)
    print(f"Baseline accuracy: {baseline_acc:.2f}%")

    results = ablation_study(model, train_loader, device, baseline_acc)

    print("\nTop blocks by importance:")
    for block, drop in sorted(results.items(), key=lambda x: x[1]):
        print(f"{block}: drop {drop:.2f}%")

    print("\n📊 Model stats before pruning:")
    print_model_flops_and_params(model)
    print(f"Params before: {count_parameters(model)/1e6:.2f}M")

    k = cfg['k_blocks_to_prune']
    candidates = [(n, d) for n, d in results.items() if not n.endswith('.0')]
    to_prune = sorted(candidates, key=lambda x: x[1])[:k]

    for block_name, drop in to_prune:
        block = dict(model.named_modules())[block_name]
        in_c, out_c = get_block_io_channels(block)
        print(f"Pruning {block_name} ({drop:.2f}%)")
        replace_block(model, block_name, cfg['replacement_type'], in_c, out_c)

    acc = evaluate(model, val_loader, device)
    print(f"✅ Accuracy after pruning: {acc:.2f}%")
    print("\n📊 Model stats after pruning:")
    print_model_flops_and_params(model)
    print(f"Params after: {count_parameters(model)/1e6:.2f}M")

    fname = f"{cfg['model_name']}_pruned_top_{k}_{cfg['dataset']}.pt"
    torch.save(model, fname)
    print(f"✅ Pruned model saved to {fname}")

if __name__ == "__main__":
    main()
