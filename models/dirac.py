import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import dirac_

# === Normalization function ===
def normalize(w):
    """Normalizes weight tensor over full filter."""
    return F.normalize(w.view(w.shape[0], -1)).view_as(w)

# === Shared Mixin Class for Dirac Parameterization ===
class DiracConv(nn.Module):
    def init_params(self, out_channels):
        self.alpha = nn.Parameter(torch.ones(out_channels))
        self.beta = nn.Parameter(torch.full((out_channels,), 0.1))
        self.register_buffer('delta', dirac_(self.weight.data.clone()))
        assert self.delta.shape == self.weight.shape
        self.v = (-1,) + (1,) * (self.weight.dim() - 1)

    def transform_weight(self):
        return self.alpha.view(*self.v) * self.delta + self.beta.view(*self.v) * normalize(self.weight)


# === Dirac Conv2D layer ===
class DiracConv2d(nn.Conv2d, DiracConv):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1, dilation=1, bias=False):
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, bias=bias)
        self.init_params(out_channels)

    def forward(self, x):
        return F.conv2d(x, self.transform_weight(), self.bias, self.stride, self.padding, self.dilation)


# === DiracBasicBlock ===
class DiracBasicBlock(nn.Module):
    expansion = 1  # Same as standard BasicBlock

    def __init__(self, in_channels, out_channels, stride=1):
        super(DiracBasicBlock, self).__init__()
        self.conv1 = DiracConv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = DiracConv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        return out


class DiracBottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1):
        super(DiracBottleneck, self).__init__()

        self.conv1 = DiracConv2d(in_channels, out_channels, kernel_size=1, padding=0)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = DiracConv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.conv3 = DiracConv2d(out_channels, out_channels * self.expansion, kernel_size=1, padding=0)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.relu(self.bn3(self.conv3(out)))
        return out



