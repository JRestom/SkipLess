import os
import yaml
import wandb
import torch
import random
import numpy as np
import argparse
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision import transforms
from torchvision.datasets import CIFAR10, CIFAR100, ImageFolder
from models import resnet50, resnet50bn, vit_base, resnet18, resnet34, dirac18, dirac34, dirac50, resnet101, resnet152
from utils import compute_stage_grad_norms
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
from tqdm import tqdm
import math



def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = os.environ.get('MASTER_PORT', '29500')
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    print(f"[Rank {rank}] Visible CUDA devices: {torch.cuda.device_count()}, using device {torch.cuda.current_device()}")

def cleanup():
    dist.destroy_process_group()

def main_worker(rank, world_size, args):
    setup(rank, world_size)

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Set seed
    device = torch.device(f"cuda:{rank}")
    seed = config['training']['seed']
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Transforms
    dataset_name = config['training']['dataset']

    if dataset_name == 'imagenet':
        train_transforms = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        test_transforms = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    else:
        train_transforms = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
        ])
        test_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
        ])

    
    if dataset_name == 'cifar10':
        train_dataset = CIFAR10(root="./data_cifar", train=True, download=True, transform=train_transforms)
        test_dataset = CIFAR10(root="./data_cifar", train=False, download=True, transform=test_transforms)
    elif dataset_name == 'cifar100':
        train_dataset = CIFAR100(root="./data_cifar100", train=True, download=True, transform=train_transforms)
        test_dataset = CIFAR100(root="./data_cifar100", train=False, download=True, transform=test_transforms)
    elif dataset_name == 'imagenet':
        train_dataset = ImageFolder(root=config['data']['imagenet_train_path'], transform=train_transforms)
        test_dataset = ImageFolder(root=config['data']['imagenet_val_path'], transform=test_transforms)

    else:
        raise ValueError("Unsupported dataset")

    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        sampler=train_sampler,
        num_workers=16,
        pin_memory=True
    )

    # Evaluation is only done on rank 0 using the full test set
    if rank == 0:
        test_loader = DataLoader(
            test_dataset,
            batch_size=config['training']['batch_size'],
            shuffle=False,  # no shuffle for evaluation
            num_workers=4,
            pin_memory=True
        )
    else:
        test_loader = None


    # Hyperparameters
    model_name = config['training']['model']
    num_classes = config['training']['num_classes']
    scheduler_type = config['model_params']['skip_scheduler']
    epochs = config['training']['epochs']
    final_skip_values = config['model_params'].get('final_skip_values', [1.0]*4)
    start_value = config['model_params'].get('start_value', None)
    update_per_batch = config['model_params'].get('update_per_batch', False)
    min_bitwidth = config['model_params'].get('min_bitwidth', 32)
    enable_quantization = config['training'].get('enable_quantization', False)
    skip_scalar = config['model_params'].get('skip_scalar', 1.0)
    final_skip_epoch = config['model_params'].get('final_skip_epoch', epochs)
    


    wandb_config = {
        "batch_size": config['training']['batch_size'],
        "epochs": config['training']['epochs'],
        "lr": config['training']['lr'],
        "weight_decay": config['training']['weight_decay'],
        "model": model_name,
        "scheduler_type": scheduler_type,
        "final_skip_values": final_skip_values,
        "enable_quantization": enable_quantization,
        "min_bitwidth": min_bitwidth,
        "final_skip_epoch": final_skip_epoch,
        "skip_scalar_vit": skip_scalar
    }

    if rank == 0 and config['wandb']['enable']:
        wandb.init(
            project=config['wandb']['project'],
            entity=config['wandb']['entity'],
            name=config['wandb']['run_name'],
            group=config['wandb'].get('group', "default"),
            config=config
        )

    # Initialized model
    if model_name == 'resnet50':
        # model = resnet50(num_classes=num_classes, scheduler_type=scheduler_type, total_epochs=epochs,
        #                  final_skip_values=final_skip_values, start_value=start_value,
        #                  min_bitwidth=min_bitwidth, enable_quantization=enable_quantization)

        model_path = config['training']['pretrained_weights']
        assert model_path.endswith('.pt'), "Expected full model file (.pt), not state_dict (.pth)"
        map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}
        model = torch.load(model_path, map_location=map_location)
        model = model.to(device)
        print(model)

   
    
    elif model_name == 'resnet18':
        model = resnet18(num_classes=num_classes, scheduler_type=scheduler_type, total_epochs=epochs,
                         final_skip_values=final_skip_values, start_value=start_value,
                         min_bitwidth=min_bitwidth, enable_quantization=enable_quantization)
    elif model_name == 'resnet34':
        model = resnet34(num_classes=num_classes, scheduler_type=scheduler_type, total_epochs=epochs,
                         final_skip_values=final_skip_values, start_value=start_value,
                         min_bitwidth=min_bitwidth, enable_quantization=enable_quantization)
    elif model_name == 'vit_base':
        model = vit_base(num_classes=num_classes, img_size=32, patch=8, hidden=384, num_layers=7, head=8,
                         dropout=0, is_cls_token=True, skip_scalar=skip_scalar, start_value=start_value,
                         scheduler_type=scheduler_type, total_epochs=epochs, final_skip=final_skip_values[0])

    elif model_name == 'dirac18':
        model = dirac18(num_classes=num_classes)

    elif model_name == 'dirac34':
        model = dirac34(num_classes=num_classes)

    elif model_name == 'dirac50':
        model = dirac50(num_classes=num_classes)

    elif model_name == 'resnet101':
        model = resnet101(num_classes=num_classes, scheduler_type=scheduler_type, total_epochs=epochs,
                          final_skip_values=final_skip_values, start_value=start_value,
                          min_bitwidth=min_bitwidth, enable_quantization=enable_quantization)

    elif model_name == 'resnet152':
        model = resnet152(num_classes=num_classes, scheduler_type=scheduler_type, total_epochs=epochs,
                          final_skip_values=final_skip_values, start_value=start_value,
                          min_bitwidth=min_bitwidth, enable_quantization=enable_quantization)
        
    else:
        raise ValueError("Unsupported model")

    model = model.to(device)
    pretrained_weights = config['training'].get('pretrained_weights', None)

    # if pretrained_weights:
    #     map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}  # maps weights properly for DDP
    #     model.load_state_dict(torch.load(pretrained_weights, map_location=map_location))

    #     if rank == 0:
    #         print(f"✅ Loaded pretrained weights from {pretrained_weights}")

    

    model = DDP(model, device_ids=[rank])

    base_lr = config['training']['lr']
    per_gpu_batch_size = config['training']['batch_size']
    scaled_lr = base_lr * (per_gpu_batch_size * 4) / 256
    config['training']['lr'] = scaled_lr

    
    optimizer = optim.AdamW(model.parameters(), lr=config['training']['lr'], weight_decay=config['training']['weight_decay'])

    total_steps = len(train_loader) * epochs
    warmup_steps = int(0.1 * total_steps)

    lrs = torch.tensor([
        config['training']['lr'] * step / warmup_steps if step < warmup_steps else
        0.5 * config['training']['lr'] * (1 + math.cos(math.pi * (step - warmup_steps) / (total_steps - warmup_steps)))
        for step in range(total_steps)
    ], dtype=torch.float32)

    # lrs = torch.cat([
    #     torch.linspace(0, config['training']['lr'], warmup_steps),
    #     torch.linspace(config['training']['lr'], 0, total_steps - warmup_steps)
    # ])

    # optimizer = torch.optim.SGD(
    # model.parameters(), 
    # lr=config['training']['lr'], 
    # momentum=0.9, 
    # weight_decay=config['training']['weight_decay'])

    #warmup_epochs = int(0.1 * config['training']['epochs'])  # e.g., 10% of total
    #main_epochs = config['training']['epochs'] - warmup_epochs

    # Warmup: Linear increase from 0 -> base LR
    #warmup_scheduler = LinearLR(optimizer, start_factor=1e-3, end_factor=1.0, total_iters=warmup_epochs)

    # Main: Cosine decay
    #cosine_scheduler = CosineAnnealingLR(optimizer, T_max=main_epochs)

    # Combine both into a sequential scheduler
    #scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_epochs])

    # total_steps = len(train_loader) * epochs
    # warmup_steps = int(0.1 * total_steps)
    # lrs = torch.cat([
    #     torch.linspace(0, config['training']['lr'], warmup_steps),
    #     torch.linspace(config['training']['lr'], 0, total_steps - warmup_steps)
    # ])

    for epoch in range(1, epochs + 1):
        train_sampler.set_epoch(epoch)
        model.train()

        if not update_per_batch:
            pass
            #model.module.update_skip_scale(epoch, total_steps=final_skip_epoch)

        running_loss, total_correct, total_samples = 0.0, 0, 0
        epoch_stage_gradients = {f"stage{i+1}": [] for i in range(4)} if model_name != 'vit_base' else []

        total_batches = final_skip_epoch * len(train_loader)
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch} [Training]", leave=False)
        for batch_idx, (inputs, targets) in enumerate(progress_bar):
        # for batch_idx, (inputs, targets) in enumerate(train_loader):
            global_step = (epoch - 1) * len(train_loader) + batch_idx
            inputs, targets = inputs.to(device), targets.to(device)

            if update_per_batch:
                pass
                #model.module.update_skip_scale(epoch, total_steps=final_skip_epoch)

            for group in optimizer.param_groups:
                group['lr'] = lrs[global_step]

            outputs = model(inputs)
            loss = F.cross_entropy(outputs, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            total_samples += targets.size(0)
            total_correct += outputs.argmax(dim=1).eq(targets).sum().item()

            if model_name != 'vit_base':
                stage_grad_norms = compute_stage_grad_norms(model.module)
                for stage in epoch_stage_gradients:
                    epoch_stage_gradients[stage].append(stage_grad_norms[stage])
            else:
                total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach()) for p in model.parameters() if p.grad is not None]))
                epoch_stage_gradients.append(total_norm.item())

        epoch_loss = running_loss / total_samples
        epoch_acc = 100. * total_correct / total_samples

        if model_name != 'vit_base':
            avg_stage_grad_norms = {stage: sum(vals) / len(vals) for stage, vals in epoch_stage_gradients.items()}
        else:
            avg_stage_grad_norms = {"vit_avg_grad_norm": sum(epoch_stage_gradients) / len(epoch_stage_gradients)}

        if rank == 0:
            print(f"Epoch {epoch}: Train Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%")
            if config['wandb']['enable']:
                wandb.log({
                    "epoch": epoch,
                    "train_loss": epoch_loss,
                    "train_accuracy": epoch_acc,
                    **avg_stage_grad_norms,
                    **{f"skip_scale/{stage}": scale for stage, scale in model.module.get_skip_scales().items()}
                })

        #scheduler.step()

        # Evaluation on one rank only
        if rank == 0:
            model.eval()
            val_loss, val_correct = 0.0, 0
            with torch.no_grad():
                for inputs, targets in test_loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = model(inputs)
                    loss = F.cross_entropy(outputs, targets)
                    val_loss += loss.item() * inputs.size(0)
                    val_correct += outputs.argmax(dim=1).eq(targets).sum().item()

            val_loss /= len(test_loader.dataset)
            val_acc = 100. * val_correct / len(test_loader.dataset)

            print(f"Validation Loss: {val_loss:.4f}, Accuracy: {val_acc:.2f}%")
            if config['wandb']['enable']:
                wandb.log({"val_loss": val_loss, "val_accuracy": val_acc})

    if rank == 0 and config.get("training", {}).get("save_weights", False):
        save_path = f"{config['wandb']['run_name']}.pth"
        torch.save(model.module.state_dict(), save_path)
        print(f"✅ Model weights saved to {save_path}")

    cleanup()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()

    world_size = torch.cuda.device_count()
    mp.spawn(main_worker, args=(world_size, args), nprocs=world_size, join=True)

if __name__ == '__main__':
    main()