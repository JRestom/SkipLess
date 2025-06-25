import os
import math
import pickle
from collections import defaultdict
import yaml

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from tqdm import tqdm
import json

import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

from models import resnet50

# ------------------ Load YAML Config ------------------
with open('/home/jose/SkipLess/yaml/imprinting/config_imprinting_ranking.yml', 'r') as f:
    config = yaml.safe_load(f)

data_dir = config['data']
batch_size = config['batch_size']
save_path = config['save_path']
imprinting_path = config['imprinting_weights']
target_embedding_size = config['target_embedding_size']
checkpoint_path = config['checkpoint_path']

device = 'cuda' if torch.cuda.is_available() else 'cpu'
num_classes = 1000

# ------------------ Dataset (Validation set) ------------------
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    normalize
])

val_dataset = datasets.ImageFolder(os.path.join(data_dir, 'val'), transform=transform)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=16, pin_memory=True)

# ------------------ Model ------------------
model = resnet50(num_classes=num_classes)
model.load_state_dict(torch.load(checkpoint_path), strict=True)
model.to(device)
model.eval()

# ------------------ Load Imprinted Weights ------------------
with open(imprinting_path, 'rb') as f:
    imprinting_data = pickle.load(f)

layer_names = imprinting_data['layer_names']
imprinted_weights = imprinting_data['weights']

# ------------------ Hook Functions ------------------
activations = {}

def l2_norm_and_pool(output):
    B, C, H, W = output.shape
    dim = round(math.sqrt(target_embedding_size / C))
    pooled = F.adaptive_avg_pool2d(output, (dim, dim)).view(B, -1)
    return pooled

def forward_hook(name):
    def hook(module, input, output):
        pooled = l2_norm_and_pool(output)
        if name not in activations:
            activations[name] = []
        activations[name].append(pooled.detach().cpu())
    return hook

# ------------------ Register Hooks ------------------
for name, module in model.named_modules():
    if name in layer_names:
        module.register_forward_hook(forward_hook(name))

# ------------------ Evaluation ------------------
correct_per_layer = {name: 0 for name in layer_names}
total_samples = 0

print("Evaluating per-layer accuracy...")
for imgs, labels in tqdm(val_loader):
    imgs, labels = imgs.to(device), labels.to(device)
    with torch.no_grad():
        model(imgs)

    for name in layer_names:
        feats = torch.cat(activations[name], dim=0)
        weights = imprinted_weights[name]  # shape: [1000, D]
        scores = torch.matmul(feats, weights.T.cpu())  # [B, 1000]
        preds = scores.argmax(dim=1).to(labels.device)
        correct = (preds == labels).sum().item()
        correct_per_layer[name] += correct

    total_samples += labels.size(0)
    activations.clear()

# ------------------ Compute Accuracy and Ranking ------------------
layer_accuracy = {name: correct_per_layer[name] / total_samples for name in layer_names}
sorted_ranking = sorted(layer_accuracy.items(), key=lambda x: x[1])

# ------------------ Save Result ------------------
# Convert ranking to a simple list for JSON
json_ranking = [{'layer': name, 'accuracy': acc} for name, acc in sorted_ranking]

# Save to JSON
os.makedirs(os.path.dirname(save_path), exist_ok=True)
with open(save_path, 'w') as f:
    json.dump({
        'layer_accuracy': layer_accuracy,
        'sorted_ranking': json_ranking
    }, f, indent=4)

print(f"Saved ranking to {save_path}")