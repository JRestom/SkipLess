import os
import math
import sys
import yaml
import json
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
from copy import deepcopy
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision import transforms, datasets
from torch.optim import AdamW
from tqdm import tqdm
from models import resnet50, resnet101, resnet152, resnet34, resnet18
from prune_utils import replace_block, get_block_io_channels, Residual1x1

def print_block_param_stats(original_block, replacement_type, in_c, out_c, rank):
    if rank != 0:
        return  # Only print from rank 0

    def describe_conv(conv, name):
        if isinstance(conv, nn.Conv2d):
            print(f"  {name}: in={conv.in_channels}, out={conv.out_channels}, kernel={conv.kernel_size}, params={sum(p.numel() for p in conv.parameters())}")
        else:
            print(f"  {name}: {type(conv).__name__}")

    print("🔍 Original Block:")
    for name, module in original_block.named_modules():
        if isinstance(module, nn.Conv2d):
            describe_conv(module, name)

    print("🔁 Replacement Block Preview:")
    if replacement_type == "conv1x1":
        conv = nn.Conv2d(in_c, out_c, kernel_size=1)
        describe_conv(conv, "conv1x1")
    elif replacement_type == "residual1x1":
        conv = nn.Conv2d(in_c, out_c, kernel_size=1)
        match_conv = nn.Identity() if in_c == out_c else nn.Conv2d(in_c, out_c, kernel_size=1)
        describe_conv(conv, "residual.conv")
        describe_conv(match_conv, "residual.match_dims")
    elif replacement_type == "identity":
        print("  Identity block has 0 parameters.")

    print("-" * 60)

class TeeLogger:
    def __init__(self, filepath):
        self.terminal = sys.stdout
        self.log = open(filepath, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()


class FeatureExtractor(nn.Module):
    def __init__(self, model, block_names):
        super().__init__()
        self.model = model
        self.block_names = block_names
        self.features = {}
        for name, module in self.model.named_modules():
            if name in self.block_names:
                module.register_forward_hook(self.save_hook(name))

    def save_hook(self, name):
        def hook(module, input, output):
            self.features[name] = output
        return hook

    def forward(self, x):
        self.features = {}
        logits = self.model(x)
        return {"logits": logits, "features": self.features}


# === Dataset Loader ===
def get_dataloaders(dataset_name, batch_size=256, rank=0, world_size=1):
    if dataset_name == "imagenet":
        data_root = '/share/data/drive_1/talal_datasets/imagenet/images'
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        val_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        train_dataset = datasets.ImageFolder(os.path.join(data_root, "train"), transform=train_transform)
        val_dataset = datasets.ImageFolder(os.path.join(data_root, "val"), transform=val_transform)

    elif dataset_name == "cifar10":
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
        ])
        val_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
        ])
        train_dataset = datasets.CIFAR10(root='./data_cifar', train=True, download=True, transform=train_transform)
        val_dataset = datasets.CIFAR10(root='./data_cifar', train=False, download=True, transform=val_transform)

    else:
        raise ValueError("Unsupported dataset")

    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    return train_loader, val_loader



def distillation_loss(student_features, teacher_features, logits, labels, alpha=1.0, beta=1.0):
    mse = nn.MSELoss()
    feature_loss = 0.0
    for name in student_features:
        sf = student_features[name]
        tf = teacher_features[name]

        if sf.shape != tf.shape:
            tf = F.interpolate(tf, size=sf.shape[2:], mode='bilinear', align_corners=False)
        feature_loss += mse(sf, tf)

    ce_loss = 0.0
    if beta > 0 and labels is not None:
        ce_loss = F.cross_entropy(logits, labels)

    return alpha * feature_loss + beta * ce_loss



# === Setup/Teardown for DDP ===
def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    dist.destroy_process_group()

def evaluate(model, dataloader, device):
    model.eval()
    total, correct = 0, 0
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            preds = model(x).argmax(dim=1)
            correct += preds.eq(y).sum().item()
            total += y.size(0)
    return 100 * correct / total

def count_total_parameters(model):
    return sum(p.numel() for p in model.parameters())

def distill_worker(rank, world_size, config):
    setup(rank, world_size)
    device = torch.device(f"cuda:{rank}")

    if rank == 0:
        log_path = config['log_path']
        sys.stdout = TeeLogger(log_path)
        print(f"📓 Logging distillation + finetuning to: {log_path}")

    model_factory = {
        'resnet18': resnet18,
        'resnet34': resnet34,
        'resnet50': resnet50,
        'resnet101': resnet101,
        'resnet152': resnet152
    }

    teacher = model_factory[config['models_name']](num_classes=1000).to(device)
    teacher.load_state_dict(torch.load(config['teacher_path'], map_location={"cuda:0": f"cuda:{rank}"}))
    teacher.eval()

    student = deepcopy(teacher)

    # === Replace Blocks ===
    with open(config['results_json'], 'r') as f:
        results = json.load(f)

    original_params = count_total_parameters(student)
    print(f"[Rank {rank}] 🔢 Total parameters before replacement: {original_params / 1e6:.2f}M")

    replaced_blocks = []
    for block_name, drop in results.items():
        if drop < config['threshold']:
            block = dict(student.named_modules())[block_name]
            in_c, out_c = get_block_io_channels(block)

            print_block_param_stats(block, config['replacement_type'], in_c, out_c, rank)  # 👈 pass `rank`

            print(f"[Rank {rank}] Replacing {block_name} with {config['replacement_type']}")
            replace_block(student, block_name, config['replacement_type'], in_c, out_c)
            replaced_blocks.append(block_name)

    if rank == 0:
        print("🔁 Replaced blocks:", replaced_blocks)


    # After block replacement
    updated_params = count_total_parameters(student)
    print(f"[Rank {rank}] 🪄 Total parameters after replacement: {updated_params / 1e6:.2f}M")

    # Report savings
    savings = 100.0 * (original_params - updated_params) / original_params
    print(f"[Rank {rank}] 💡 Total parameter reduction: {savings:.2f}%")

    # Freeze all layers except the replaced ones
    for name, param in student.named_parameters():
        param.requires_grad = any(name.startswith(b) for b in replaced_blocks)

    print("Trainable parameters:")
    for name, param in student.named_parameters():
        if param.requires_grad:
            print(f"  ✅ {name}")

    student = FeatureExtractor(student.to(device), replaced_blocks)
    teacher = FeatureExtractor(teacher.to(device), replaced_blocks)

    student = DDP(student, device_ids=[rank])
    train_loader, val_loader = get_dataloaders(config['dataset'], batch_size=256, rank=rank, world_size=world_size)

    base_lr = config['lr']
    #scaled_lr = base_lr * world_size
    optimizer = AdamW(filter(lambda p: p.requires_grad, student.parameters()), lr=base_lr)
    # if rank == 0:
    #     print(f"📈 Adjusted learning rate for {world_size} GPUs: {scaled_lr}")

    # === Distillation Training Loop ===
    for epoch in range(config['epochs_distill']):
        train_loader.sampler.set_epoch(epoch)
        student.train()
        total_loss = 0.0
        for x, y in tqdm(train_loader, desc=f"[GPU {rank}] Epoch {epoch+1}", disable=(rank != 0)):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            s_out = student(x)
            with torch.no_grad():
                t_out = teacher(x)
            loss = distillation_loss(s_out['features'], t_out['features'], s_out['logits'], y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if rank == 0:
            print(f"Epoch {epoch+1} | Loss: {total_loss:.4f}")
            acc = evaluate(student.module.model, val_loader, device)
            print(f"✅Validation accuracy: {acc:.2f}%")

    if rank == 0:
        torch.save(student.module.model.state_dict(), config['save_path_distill'])
        print(f"✅ Distilled model saved to {config['save_path_distill']}")

    # === FINETUNING PHASE ===
    if rank == 0:
        print(f"🔧 Starting finetuning...")
    
    student.module.model.requires_grad_(True)

    # Count trainable parameters
    if rank == 0:
        num_trainable_params = sum(p.numel() for p in student.module.model.parameters() if p.requires_grad)
        print(f"[Rank {rank}] 🔍 Number of trainable parameters after unfreezing: {num_trainable_params / 1e6:.2f}M")

    student = DDP(student.module.model.to(device), device_ids=[rank])

    train_sampler = DistributedSampler(train_loader.dataset, num_replicas=world_size, rank=rank, shuffle=True)
    train_loader = DataLoader(train_loader.dataset, batch_size=config['batch_size'], sampler=train_sampler, num_workers=16, pin_memory=True)

    base_lr = config['lr_ft']
    scaled_lr = base_lr * world_size
    optimizer = AdamW(student.parameters(), lr=scaled_lr, weight_decay=0.1)

    # LR scheduler
    steps_per_epoch = len(train_loader)
    total_steps = config['epochs_finetune'] * steps_per_epoch
    warmup_steps = int(0.1 * total_steps)

    lrs = torch.tensor([
    scaled_lr * step / warmup_steps if step < warmup_steps else
    0.5 * scaled_lr * (1 + math.cos(math.pi * (step - warmup_steps) / (total_steps - warmup_steps)))
    for step in range(total_steps) ], dtype=torch.float32)

    criterion = nn.CrossEntropyLoss()

    for epoch in range(config['epochs_finetune']):
        train_sampler.set_epoch(epoch)
        student.train()

        running_loss, correct, total = 0.0, 0, 0
        progress = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"[GPU {rank}] Finetune Epoch {epoch+1}", disable=(rank != 0))
        
        for batch_idx, (x, y) in progress:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            outputs = student(x)
            loss = criterion(outputs, y)

            global_step = epoch * steps_per_epoch + batch_idx
            for group in optimizer.param_groups:
                group['lr'] = lrs[global_step]

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * x.size(0)
            correct += outputs.argmax(dim=1).eq(y).sum().item()
            total += y.size(0)

        if rank == 0:
            avg_loss = running_loss / total
            train_acc = 100. * correct / total
            val_acc = evaluate(student.module, val_loader, device)
            print(f"[Finetune Epoch {epoch+1}] Loss: {avg_loss:.4f} | Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")

    if rank == 0:
        torch.save(student.module.state_dict(), config['save_path_ft'])
        print(f"✅ Final model saved to {config['save_path_ft']}")

    cleanup()

def finetune_ddp(student, train_dataset, val_loader, device, rank, world_size, epochs=30, lr=1e-4, batch_size=256):
    print(f"[Rank {rank}] 🔧 Starting finetuning...")

    # Unfreeze all layers
    for param in student.parameters():
        param.requires_grad = True

    student.to(device)
    student = DDP(student, device_ids=[rank])

    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler,
                              num_workers=16, pin_memory=True)

    optimizer = AdamW(student.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        train_sampler.set_epoch(epoch)
        student.train()

        running_loss = 0.0
        correct = 0
        total = 0

        if rank == 0:
            progress = tqdm(train_loader, desc=f"[Finetune] Epoch {epoch+1}")
        else:
            progress = train_loader  # No tqdm on other ranks

        for x, y in progress:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            outputs = student(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * x.size(0)
            correct += outputs.argmax(dim=1).eq(y).sum().item()
            total += y.size(0)


        if rank == 0:
            train_acc = 100. * correct / total
            avg_loss = running_loss / total
            val_acc = evaluate(student.module, val_loader, device)
            print(f"[Finetune Epoch {epoch+1}] Loss: {avg_loss:.4f} | Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")

    if rank == 0:
        print("✅ Finetuning complete.\n")
        torch.save(student.module.model.state_dict(), config['save_path_ft'])
        print(f"✅ Distilled model saved to {config['save_path_ft']}")

    

# === Entry Point ===
def main():
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--teacher_path', type=str, required=True)
    # parser.add_argument('--model_name', type=str, required=True)
    # parser.add_argument('--dataset', type=str, required=True)
    # parser.add_argument('--results_json', type=str, required=True)
    # parser.add_argument('--replacement_type', type=str, default="residual1x1")
    # parser.add_argument('--threshold', type=float, default=7.0)
    # parser.add_argument('--save_path', type=str, required=True)
    # args = parser.parse_args()

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    world_size = torch.cuda.device_count()
    mp.spawn(distill_worker, args=(world_size, config), nprocs=world_size, join=True)


if __name__ == '__main__':
    main()
