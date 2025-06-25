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

from models import resnet50

# ------------------ Load YAML Config ------------------
with open('/home/jose/SkipLess/yaml/imprinting/config_imprinting.yml', 'r') as f:
    config = yaml.safe_load(f)


data_dir = config['data']
batch_size = config['batch_size']
ratio = config['ratio']
save_path = config['save_path']
target_embedding_size = config['target_embedding_size']
checkpoint_path = config['checkpoint_path']

device = 'cuda' if torch.cuda.is_available() else 'cpu'
num_classes = 1000

# ------------------ Dataset ------------------
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    normalize
])

dataset = datasets.ImageFolder(os.path.join(data_dir, 'train'), transform=transform)
total_len = int(len(dataset) * ratio)
subset, _ = torch.utils.data.random_split(dataset, [total_len, len(dataset) - total_len])
dataloader = DataLoader(subset, batch_size=batch_size, shuffle=False, num_workers=16, pin_memory=True)

# ------------------ Model ------------------
model = resnet50(num_classes=num_classes)
model.load_state_dict(torch.load(config['checkpoint_path']), strict=True)
model.to(device)
model.eval()

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
layer_names = []
for name, module in model.named_modules():
    if isinstance(module, nn.Conv2d):
        module.register_forward_hook(forward_hook(name))
        layer_names.append(name)


# ------------------ Initialize Class Sums ------------------
class_sums = {name: defaultdict(lambda: 0) for name in layer_names}
class_counts = defaultdict(int)

# ------------------ Imprinting ------------------
print("Collecting activations...")
for imgs, labels in tqdm(dataloader):
    imgs, labels = imgs.to(device), labels.to(device)
    with torch.no_grad():
        model(imgs)

    for name in layer_names:
        feats = torch.cat(activations[name], dim=0)
        for i in range(feats.shape[0]):
            cls = labels[i].item()
            class_sums[name][cls] += feats[i]
            class_counts[cls] += 1
        activations[name] = []

# ------------------ Compute Imprinted Weights ------------------
imprinted_weights = {}

for name in layer_names:
    class_weights = []
    for cls in range(num_classes):
        if class_counts[cls] == 0:
            raise ValueError(f"Class {cls} was not sampled. Increase ratio.")
        summed = class_sums[name][cls]
        avg = summed / class_counts[cls]
        avg_norm = F.normalize(avg, dim=0)
        class_weights.append(avg_norm.unsqueeze(0))
    imprinted_weights[name] = torch.cat(class_weights, dim=0)

# ------------------ Save Result ------------------
save_obj = {
    'layer_names': layer_names,
    'weights': imprinted_weights
}
os.makedirs(os.path.dirname(save_path), exist_ok=True)
with open(save_path, 'wb') as f:
    pickle.dump(save_obj, f)

print(f"Saved imprinted weights to {save_path}")