import torch
import os
import torch.nn as nn
import wandb
import argparse 
from models import resnet50, resnet50bn, vit_base, resnet18, resnet34, dirac18, dirac34, dirac50, resnet50_dirac_combined
from torchvision.models import resnet50 as tv_resnet50, resnet101 as tv_resnet101, resnet152 as tv_resnet152, resnet34 as tv_resnet34, resnet18 as tv_resnet18
from torchvision import transforms
from torchvision.datasets import CIFAR10, ImageFolder
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
import torch.optim as optim
from tqdm import tqdm 
import torch.backends.cudnn as cudnn
import sys
from fvcore.nn import FlopCountAnalysis, parameter_count_table

def print_model_flops_and_params(model, input_shape=(1, 3, 224, 224)):
    model.eval()
    dummy_input = torch.randn(input_shape).to(next(model.parameters()).device)
    flops = FlopCountAnalysis(model, dummy_input)
    print("🧮 FLOPs and parameter count:")
    print(parameter_count_table(model))
    print(f"Total FLOPs: {flops.total() / 1e9:.2f} GFLOPs")


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def evaluate(model, val_loader, device):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            preds = outputs.argmax(dim=1)
            correct += preds.eq(targets).sum().item()
            total += targets.size(0)

    accuracy = 100.0 * correct / total
    return accuracy


def ablation_study(model, val_loader, device, baseline_acc):

     # Count how many blocks will be ablated
    blocks = [name for name, block in model.named_modules() if hasattr(block, 'disable_main')]
    results = {}

    for name in tqdm(blocks, desc="Running ablation study"):
        block = dict(model.named_modules())[name]
        block.disable_main = True
        ablated_acc = evaluate(model, val_loader, device)
        acc_drop = baseline_acc - ablated_acc
        results[name] = acc_drop
        block.disable_main = False

    # for name, block in model.named_modules():
    #     if hasattr(block, 'disable_main'):
    #         print(f"Ablating main path at {name}")

    #         # Disable main path
    #         block.disable_main = True

    #         # Evaluate
    #         ablated_acc = evaluate(model, val_loader, device)

    #         # Record drop
    #         acc_drop = baseline_acc - ablated_acc
    #         results[name] = acc_drop

    #         # Restore block
    #         block.disable_main = False

    return results


def replace_block(model, block_name):
    parts = block_name.split('.')
    mod = model
    for p in parts[:-1]:
        mod = getattr(mod, p)
    setattr(mod, parts[-1], nn.Identity())


def get_dataloaders(dataset_name, data_root, batch_size=512):
    if dataset_name == "cifar10":
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
        #train_dataset = CIFAR10(root=data_root, train=True, download=True, transform=train_transforms)
        val_dataset = CIFAR10(root=data_root, train=False, download=True, transform=test_transforms)

    elif dataset_name == "imagenet":

        train_transforms = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        test_transforms = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        #train_dataset = ImageNet(root=data_root, split="train", transform=train_transforms)
        val_dataset = ImageFolder(root=os.path.join(data_root, "val"), transform=test_transforms)

    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=16, pin_memory=True)
    return val_loader

if __name__ == "__main__":
 
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model weights')
    parser.add_argument('--dataset', type=str, required=True, choices=["cifar10", "imagenet"], help='Dataset: cifar10 or imagenet')
    parser.add_argument('--model_name', type=str, required=True, choices=["resnet18", "resnet34", "resnet50", "resnet101", "resnet152"], help='ResNet model variant')
    args = parser.parse_args()

    #log_file = open('pruning_log.txt', 'w')
    #sys.stdout = log_file
    
    cudnn.benchmark = True
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    if args.model_name == "resnet50":
        model = resnet50(num_classes=1000)
        model.load_state_dict(torch.load(args.checkpoint))
        

    elif args.model_name == "resnet101":
        model = tv_resnet101(pretrained=True)

    elif args.model_name == "resnet152":
        model = tv_resnet152(pretrained=True)
        
    else:
        raise ValueError(f"Unsupported model: {args.model_name}")

    
    model.to(device)
    
    
    
    data_root='/l/users/jose.viera/datasets/ILSVRC/Data/CLS-LOC'
    #data_root='/home/jose.viera/projects/partialSkip/removing_skips/data_cifar'
    val_loader = get_dataloaders(args.dataset, data_root=data_root)

    # Baseline acc
    baseline_acc = evaluate(model, val_loader, device)
    print(f"Baseline accuracy: {baseline_acc:.2f}%")

    # Run ablation
    results = ablation_study(model, val_loader, device, baseline_acc)


    # Optional: print
    for block, drop in results.items():
        print(f"{block}: Accuracy drop {drop:.2f}%")

    # Before pruning

    print("📊 Before pruning:")
    print_model_flops_and_params(model)

    param_count_before = count_parameters(model)
    # print(f"Total parameters before pruning: {param_count_before / 1e6:.2f}M")


    
    # === Pruning based on threshold ===
    threshold = 7.0  # Example: replace if less than 1.0% accuracy drop
    for block, drop in results.items():
        if drop < threshold:
            print(f"Pruning {block} (acc drop {drop:.2f}%)")
            replace_block(model, block)

    pruned_accuracy = evaluate(model, val_loader, device)
    print(f"✅ Accuracy after pruning: {pruned_accuracy:.2f}%")

    # After pruning

    print("📊 After pruning:")
    print_model_flops_and_params(model)

    param_count_after = count_parameters(model)
    # print(f"Total parameters after pruning: {param_count_after / 1e6:.2f}M")

    # How much we saved
    savings = 100.0 * (param_count_before - param_count_after) / param_count_before
    print(f"✅ Parameters reduced by {savings:.2f}%")

    #torch.save(model, "resnet50_pruned_7.0.pt")
    #print("✅ Full pruned model saved to resnet50_pruned.pt")


    # fine_tune_epochs = 20
    # #weight_decay = 0.05 
    # fine_tune_lr = 0.0001  # 10x smaller than original
    # optimizer = optim.AdamW(model.parameters(), lr=fine_tune_lr)

    # criterion = nn.CrossEntropyLoss()

    # for epoch in range(fine_tune_epochs):
    #     model.train()
    #     running_loss = 0.0
    #     total = 0
    #     correct = 0

    #     for inputs, targets in train_loader:
    #         inputs, targets = inputs.to(device), targets.to(device)

    #         optimizer.zero_grad()
    #         outputs = model(inputs)
    #         loss = criterion(outputs, targets)
    #         loss.backward()
    #         optimizer.step()

    #         running_loss += loss.item() * inputs.size(0)
    #         _, preds = outputs.max(1)
    #         correct += preds.eq(targets).sum().item()
    #         total += targets.size(0)

    #     train_acc = 100. * correct / total
    #     val_acc = evaluate(model, val_loader, device)

    #     print(f"[Fine-tuning] Epoch {epoch+1}/{fine_tune_epochs} | Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")


    


    