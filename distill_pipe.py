import os
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


# def get_dataloaders(dataset_name, batch_size=256):
#     if dataset_name == "imagenet":
#         transform = transforms.Compose([
#             transforms.Resize(256),
#             transforms.CenterCrop(224),
#             transforms.ToTensor(),
#             transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#         ])
#         data_root='/share/data/drive_1/talal_datasets/imagenet/images'
#         val_dataset = datasets.ImageFolder(os.path.join(data_root, "val"), transform=transform)
#         train_dataset = datasets.ImageFolder(os.path.join(data_root, "train"), transform=transform)

#     elif dataset_name == "cifar10":
#         transform = transforms.Compose([
#             transforms.ToTensor(),
#             transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
#         ])
#         val_dataset = datasets.CIFAR10(root=data_root, train=False, download=True, transform=transform)
#     else:
#         raise ValueError("Unsupported dataset")

#     train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=16, pin_memory=True)
#     val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=16, pin_memory=True)

#     return train_loader, val_loader


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

def distill_worker(rank, world_size, args):
    setup(rank, world_size)
    device = torch.device(f"cuda:{rank}")

    model_factory = {
        'resnet18': resnet18,
        'resnet34': resnet34,
        'resnet50': resnet50,
        'resnet101': resnet101,
        'resnet152': resnet152
    }

    teacher = model_factory[args.model_name](num_classes=1000).to(device)
    teacher.load_state_dict(torch.load(args.teacher_path, map_location={"cuda:0": f"cuda:{rank}"}))
    teacher.eval()

    student = deepcopy(teacher)

    # === Replace Blocks ===
    with open(args.results_json, 'r') as f:
        results = json.load(f)

    replaced_blocks = []
    for block_name, drop in results.items():
        if drop < args.threshold:
            block = dict(student.named_modules())[block_name]
            in_c, out_c = get_block_io_channels(block)
            print(f"Replacing {block_name} with {args.replacement_type}")
            replace_block(student, block_name, args.replacement_type, in_c, out_c)
            replaced_blocks.append(block_name)

    print("🔁 Replaced blocks:", replaced_blocks)

    if not replaced_blocks:
        raise ValueError("No blocks were replaced. Cannot distill.")

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
    train_loader, val_loader = get_dataloaders(args.dataset, batch_size=256, rank=rank, world_size=world_size)

    base_lr = 1e-4
    scaled_lr = base_lr * world_size
    optimizer = AdamW(filter(lambda p: p.requires_grad, student.parameters()), lr=scaled_lr)
    if rank == 0:
        print(f"📈 Adjusted learning rate for {world_size} GPUs: {scaled_lr}")

    # # === Print trainable layers ===
    # if rank == 0:
    #     print("Trainable layers:")
    #     for name, p in student.module.model.named_parameters():
    #         if p.requires_grad:
    #             print(f"  {name}")

    # === Distillation Training Loop ===
    for epoch in range(10):
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
        torch.save(student.module.model.state_dict(), args.save_path)
        print(f"✅ Distilled model saved to {args.save_path}")

    cleanup()

# === Entry Point ===
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--teacher_path', type=str, required=True)
    parser.add_argument('--model_name', type=str, required=True)
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--results_json', type=str, required=True)
    parser.add_argument('--replacement_type', type=str, default="residual1x1")
    parser.add_argument('--threshold', type=float, default=7.0)
    parser.add_argument('--save_path', type=str, required=True)
    args = parser.parse_args()

    world_size = torch.cuda.device_count()
    mp.spawn(distill_worker, args=(world_size, args), nprocs=world_size, join=True)

if __name__ == '__main__':
    main()

# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--teacher_path', type=str, required=True)
#     parser.add_argument('--model_name', type=str, required=True)
#     parser.add_argument('--dataset', type=str, required=True)
#     parser.add_argument('--results_json', type=str, required=True)
#     parser.add_argument('--replacement_type', type=str, default="residual1x1")
#     parser.add_argument('--threshold', type=float, default=7.0)
#     parser.add_argument('--save_path', type=str, required=True)
#     args = parser.parse_args()

#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#     model_factory = {
#         'resnet18': resnet18,
#         'resnet34': resnet34,
#         'resnet50': resnet50,
#         'resnet101': resnet101,
#         'resnet152': resnet152
#     }

#     teacher = model_factory[args.model_name](num_classes=1000).to(device)
#     teacher.load_state_dict(torch.load(args.teacher_path))
#     teacher.eval()

#     student = deepcopy(teacher)

#     # Load results
#     with open(args.results_json, 'r') as f:
#         results = json.load(f)

#     replaced_blocks = []
#     for block_name, drop in results.items():
#         if drop < args.threshold:
#             block = dict(student.named_modules())[block_name]
#             in_c, out_c = get_block_io_channels(block)
#             print(f"Replacing {block_name} with {args.replacement_type}")
#             replace_block(student, block_name, args.replacement_type, in_c, out_c)
#             replaced_blocks.append(block_name)

#     print("🔁 Replaced blocks:", replaced_blocks)

#     # Freeze all parameters except the replaced blocks
#     for name, param in student.named_parameters():
#         param.requires_grad = any(name.startswith(b) for b in replaced_blocks)

#     student = FeatureExtractor(student.to(device), replaced_blocks)
#     teacher = FeatureExtractor(teacher.to(device), replaced_blocks)

#     train_loader, val_loader = get_dataloaders(args.dataset)
#     optimizer = AdamW(filter(lambda p: p.requires_grad, student.parameters()), lr=1e-4)

#     print("📋 Trainable parameters:")
#     for name, param in student.model.named_parameters():
#         if param.requires_grad:
#             print(f"  ✅ {name}")

#     print("📚 Starting distillation training...")
#     student.train()
#     for epoch in range(10):
#         total_loss = 0.0
#         for x, y in tqdm(train_loader):
#             x, y = x.to(device), y.to(device)
#             optimizer.zero_grad()
#             s_out = student(x)
#             with torch.no_grad():
#                 t_out = teacher(x)
#             loss = distillation_loss(s_out['features'], t_out['features'], s_out['logits'], y, alpha=1.0, beta=1.0)
#             loss.backward()
#             optimizer.step()
#             total_loss += loss.item()
#         print(f"Epoch {epoch+1}: Loss = {total_loss:.4f}")
#         acc = evaluate(student.model, val_loader, device)
#         print(f"Validation accuracy: {acc:.2f}%")

    
#     torch.save(student.model.state_dict(), args.save_path)
#     print(f"✅ Distilled model saved to {args.save_path}")

# if __name__ == '__main__':
#     main()