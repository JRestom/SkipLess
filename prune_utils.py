import torch.nn as nn
import math

class ResidualBottleneck1x1(nn.Module):
    def __init__(self, in_channels, out_channels, bottleneck_dim=128):
        super().__init__()
        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels, bottleneck_dim, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(bottleneck_dim, out_channels, kernel_size=1, bias=False)
        )

        self.match_dims = (
            nn.Identity() if in_channels == out_channels
            else nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        )

    def forward(self, x):
        return self.bottleneck(x) + self.match_dims(x)

class Residual1x1(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.match_dims = nn.Identity() if in_channels == out_channels else nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x):
        return self.conv(x) + self.match_dims(x)


def print_block_param_stats(block, name, replacement_type=None):
    if replacement_type is None:
        print("🔍 Original Block:")
    else:
        print("🔁 Replacement Block Preview:")

    if isinstance(block, nn.Identity):
        print(f"  [Identity] No parameters.")
        return

    # Original ResNet Bottleneck block
    if hasattr(block, 'conv1') and hasattr(block, 'conv2') and hasattr(block, 'conv3'):
        for conv_name in ['conv1', 'conv2', 'conv3']:
            conv = getattr(block, conv_name)
            in_c, out_c = conv.in_channels, conv.out_channels
            k = conv.kernel_size
            params = sum(p.numel() for p in conv.parameters())
            print(f"  {conv_name}: in={in_c}, out={out_c}, kernel={k}, params={params:,}")

    # Residual1x1
    elif isinstance(block, Residual1x1):
        conv = block.conv
        in_c, out_c = conv.in_channels, conv.out_channels
        k = conv.kernel_size
        p = sum(p.numel() for p in conv.parameters())
        print(f"  conv: in={in_c}, out={out_c}, kernel={k}, params={p:,}")
        if isinstance(block.match_dims, nn.Conv2d):
            m = block.match_dims
            print(f"  match_dims: in={m.in_channels}, out={m.out_channels}, kernel={m.kernel_size}, params={sum(p.numel() for p in m.parameters()):,}")

    # ResidualBottleneck1x1
    elif isinstance(block, ResidualBottleneck1x1):
        conv1 = block.bottleneck[0]
        conv2 = block.bottleneck[2]
        print(f"  bottleneck_reduce: in={conv1.in_channels}, out={conv1.out_channels}, kernel={conv1.kernel_size}, params={sum(p.numel() for p in conv1.parameters()):,}")
        print(f"  bottleneck_expand: in={conv2.in_channels}, out={conv2.out_channels}, kernel={conv2.kernel_size}, params={sum(p.numel() for p in conv2.parameters()):,}")
        if isinstance(block.match_dims, nn.Conv2d):
            m = block.match_dims
            print(f"  match_dims: in={m.in_channels}, out={m.out_channels}, kernel={m.kernel_size}, params={sum(p.numel() for p in m.parameters()):,}")

    else:
        total = sum(p.numel() for p in block.parameters())
        print(f"  ❓ Unknown block structure. Total parameters: {total:,}")

def replace_block(model, block_name, replacement_type="identity", in_channels=None, out_channels=None):
    parts = block_name.split('.')
    mod = model
    for p in parts[:-1]:
        mod = getattr(mod, p)

    if replacement_type == "identity":
        replacement = nn.Identity()
    elif replacement_type == "conv1x1":
        replacement = nn.Conv2d(in_channels, out_channels, kernel_size=1)
    elif replacement_type == "residual1x1":
        replacement = Residual1x1(in_channels, out_channels)
    elif replacement_type == "residual_bottleneck1x1":
        replacement = ResidualBottleneck1x1(in_channels, out_channels)  # <-- New type
    else:
        raise ValueError(f"Unsupported replacement type: {replacement_type}")

    setattr(mod, parts[-1], replacement)



def get_block_io_channels(block):
    if hasattr(block, 'conv1') and hasattr(block, 'conv3'):
        in_channels = block.conv1.in_channels
        out_channels = block.conv3.out_channels
    else:
        raise ValueError("Block does not have expected conv1 and conv3 attributes.")
    return in_channels, out_channels
