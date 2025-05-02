import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import math
from .dirac import DiracBasicBlock, DiracConv2d, DiracBottleneck
from inspect import signature


# Linear Scheduler for Skip Connection
def linear_scheduler(step, total_steps, start=1.0, end=0.0):
    return max(end, start - (start - end) * (step / total_steps))

# Cosine Scheduler for Skip Connection
def cosine_scheduler(step, total_steps, start=1.0, end=0.0):
    return end + (start - end) * 0.5 * (1 + math.cos(math.pi * step / (total_steps - 1)))

def quantize_tensor(tensor, bit_width, min_bitwidth=32):
    """Quantizes the tensor to the given bit width."""
    max_val = tensor.abs().max()
    
    # Ensure bit_width does not go below the minimum configured value
    bit_width = max(bit_width, min_bitwidth)
    
    # Compute scale factor for quantization
    scale = max_val / (2**(bit_width - 1) - 1)
    
    # Apply quantization (simulated)
    quantized = torch.round(tensor / scale) * scale
    return quantized

# Unified Bottleneck with configurable skip connection scheduler
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, downsample=None, scheduler_type='linear', total_epochs=100, final_skip=1.0, update_per_batch=False, start_value=None, min_bitwidth=32, enable_quantization=False):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.downsample = downsample
        self.skip_scale = 1  # Initialized to 1
        self.scheduler_type = scheduler_type
        self.total_epochs = total_epochs
        self.final_skip = final_skip
        self.update_per_batch = update_per_batch
        self.start_value = start_value

        self.min_bitwidth = min_bitwidth # Start with FP32 by default 
        self.quantization_bitwidth = 32
        self.enable_quantization = enable_quantization
    
    def update_skip_scale(self, step, total_steps, start_value=None):

        if start_value is None:
            start_value = self.start_value

        # Stop updating after we reached total_steps
        if step >= total_steps:
            #print(f"⚠️ Skipping update: Step {step} >= Final {total_steps}, Keeping Skip Scale at {self.skip_scale:.4f}")
            return   

        if self.scheduler_type == 'linear':
            self.skip_scale = linear_scheduler(step, total_steps, start=start_value, end=self.final_skip)
        elif self.scheduler_type == 'cosine':
            self.skip_scale = cosine_scheduler(step, total_steps, start=start_value, end=self.final_skip)
        elif self.scheduler_type == 'none':
            self.skip_scale = 1.0
        else:
            raise ValueError(f"Unsupported scheduler type: {self.scheduler_type}")

        # We allow update_skip to control the quantization of the identity
        min_bitwidth = self.min_bitwidth  # Read from YAML
        max_bitwidth = 32  # We assume we start at full precision (32-bit)
        
        self.quantization_bitwidth = max(
            min_bitwidth, 
            max_bitwidth - int((step / total_steps) * (max_bitwidth - min_bitwidth))
        )

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        if self.enable_quantization:
            identity = quantize_tensor(identity, self.quantization_bitwidth, self.min_bitwidth)

        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = F.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        # Apply scaled skip connection
        out += self.skip_scale * identity
        out = F.relu(out)

        return out


#Bottleneck with BN after skip connection 
class Bottleneck_bn(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, downsample=None, scheduler_type='linear', total_epochs=100, final_skip=1.0, update_per_batch=False, start_value=None, enable_quantization=False):
        super(Bottleneck_bn, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.bn4 = nn.BatchNorm2d(out_channels * self.expansion)
        self.downsample = downsample
        self.skip_scale = 1  # Initialized to 1
        self.scheduler_type = scheduler_type
        self.total_epochs = total_epochs
        self.final_skip = final_skip
        self.update_per_batch = update_per_batch
        self.start_value = start_value
    
    def update_skip_scale(self, epoch, total_epochs, start_value=None):
        """Updates the skip scale dynamically during training."""
        if start_value is None:
            start_value = 1.0  # Normal training start

        if self.scheduler_type == 'linear':
            self.skip_scale = linear_scheduler(epoch, total_epochs, start=start_value, end=self.final_skip)
        elif self.scheduler_type == 'cosine':
            self.skip_scale = cosine_scheduler(epoch, total_epochs, start=start_value, end=self.final_skip)
        elif self.scheduler_type == 'none':
            self.skip_scale = 1.0
        else:
            raise ValueError(f"Unsupported scheduler type: {self.scheduler_type}")


    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = F.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        # Apply scaled skip connection
        out += self.skip_scale * identity
        out = self.bn4(out)
        out = F.relu(out)

        return out

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, scheduler_type='linear', total_epochs=100, final_skip=1.0, start_value=1.0):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

        self.scheduler_type = scheduler_type
        self.total_epochs = total_epochs
        self.final_skip = final_skip
        self.start_value = start_value
        self.skip_scale = start_value
        

    def update_skip_scale(self, step, total_steps):
        if step >= total_steps:
            return

        if self.scheduler_type == 'linear':
            self.skip_scale = linear_scheduler(step, total_steps, start=self.start_value, end=self.final_skip)
        elif self.scheduler_type == 'cosine':
            self.skip_scale = cosine_scheduler(step, total_steps, start=self.start_value, end=self.final_skip)
        elif self.scheduler_type == 'none':
            self.skip_scale = 1.0
        else:
            raise ValueError(f"Unsupported scheduler type: {self.scheduler_type}")

    def forward(self, x):
        identity = x

        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += self.skip_scale * identity
        out = F.relu(out)

        return out

# Original   
# class ResNet(nn.Module):
#     def __init__(self, block, layers, num_classes=10, pretrained=False, scheduler_type='none', total_epochs=100, final_skip_values=None, use_bn=False, update_per_batch=False, start_value=None, min_bitwidth=32, enable_quantization=False):
#         super(ResNet, self).__init__()
#         self.in_channels = 64
#         self.pretrained = pretrained
#         self.scheduler_type = scheduler_type
#         self.total_epochs = total_epochs
#         self.final_skip_values = final_skip_values if final_skip_values else [1.0, 1.0, 1.0, 1.0]  # Default values
#         self.use_bn = use_bn  # Determines which bottleneck to use
#         self.update_per_batch = update_per_batch
#         self.start_value = start_value
#         self.min_bitwidth = min_bitwidth
#         self.enable_quantization = enable_quantization

        
#         self.block = Bottleneck_bn if use_bn else block # Select the correct bottleneck type

#         self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
#         self.bn1 = nn.BatchNorm2d(64)
#         self.relu = nn.ReLU(inplace=True)
#         self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

#         self.layer1 = self._make_layer(self.block, 64, layers[0], stage=0, final_skip=self.final_skip_values[0], start_value=self.start_value, min_bitwidth=self.min_bitwidth, enable_quantization=self.enable_quantization)
#         self.layer2 = self._make_layer(self.block, 128, layers[1], stride=2, stage=1, final_skip=self.final_skip_values[1], start_value=self.start_value, min_bitwidth=self.min_bitwidth, enable_quantization=self.enable_quantization)
#         self.layer3 = self._make_layer(self.block, 256, layers[2], stride=2, stage=2, final_skip=self.final_skip_values[2], start_value=self.start_value, min_bitwidth=self.min_bitwidth, enable_quantization=self.enable_quantization)
#         self.layer4 = self._make_layer(self.block, 512, layers[3], stride=2, stage=3, final_skip=self.final_skip_values[3], start_value=self.start_value, min_bitwidth=self.min_bitwidth, enable_quantization=self.enable_quantization)

#         self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
#         self.fc = nn.Linear(512 * self.block.expansion, num_classes)

#         self._initialize_weights()

#     def _make_layer(self, block, out_channels, blocks, stride=1, stage=None, final_skip=0.0, start_value=None, min_bitwidth=32, enable_quantization=False):
#         downsample = None
#         if stride != 1 or self.in_channels != out_channels * block.expansion:
#             downsample = nn.Sequential(
#                 nn.Conv2d(self.in_channels, out_channels * block.expansion, kernel_size=1, stride=stride, bias=False),
#                 nn.BatchNorm2d(out_channels * block.expansion),
#             )

#         layers = []

#         if block == BasicBlock:
#             layers.append(block(
#                 self.in_channels, out_channels, stride=stride, 
#                 scheduler_type=self.scheduler_type, total_epochs=self.total_epochs, 
#                 final_skip=final_skip, start_value=start_value
#             ))

#         else:
#             layers.append(block(self.in_channels, out_channels, stride, downsample, scheduler_type=self.scheduler_type, total_epochs=self.total_epochs, final_skip=final_skip, update_per_batch=self.update_per_batch, start_value=self.start_value, min_bitwidth=self.min_bitwidth, enable_quantization=self.enable_quantization))

#         self.in_channels = out_channels * block.expansion

        
#         for _ in range(1, blocks):

#             if block == BasicBlock:
#                 layers.append(block(
#                     self.in_channels, out_channels, stride=stride, 
#                     scheduler_type=self.scheduler_type, total_epochs=self.total_epochs, 
#                     final_skip=final_skip, start_value=start_value
#                 ))

#             else:
#                 layers.append(block(self.in_channels, out_channels, scheduler_type=self.scheduler_type, total_epochs=self.total_epochs, final_skip=final_skip, update_per_batch=self.update_per_batch, start_value=self.start_value, min_bitwidth=self.min_bitwidth, enable_quantization=self.enable_quantization))

#         return nn.Sequential(*layers)

#     def update_skip_scale(self, step, total_steps):
#         for layer in [self.layer1, self.layer2, self.layer3, self.layer4]:
#             for block in layer:
#                 block.update_skip_scale(step, total_steps)



#     def get_skip_scales(self):
#         return {
#             "stage1": self.layer1[0].skip_scale,
#             "stage2": self.layer2[0].skip_scale,
#             "stage3": self.layer3[0].skip_scale,
#             "stage4": self.layer4[0].skip_scale,
#         }

#     def get_quantization_bitwidths(self):
#         return {
#             "stage1": self.layer1[0].quantization_bitwidth,
#             "stage2": self.layer2[0].quantization_bitwidth,
#             "stage3": self.layer3[0].quantization_bitwidth,
#             "stage4": self.layer4[0].quantization_bitwidth,
#         }

#     def _initialize_weights(self):
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#             elif isinstance(m, nn.BatchNorm2d):
#                 nn.init.constant_(m.weight, 1)
#                 nn.init.constant_(m.bias, 0)
#             elif isinstance(m, nn.Linear):
#                 nn.init.normal_(m.weight, 0, 0.01)
#                 nn.init.constant_(m.bias, 0)

#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.bn1(x)
#         x = self.relu(x)
#         x = self.maxpool(x)

#         x = self.layer1(x)
#         x = self.layer2(x)
#         x = self.layer3(x)
#         x = self.layer4(x)

#         x = self.avgpool(x)
#         x = torch.flatten(x, 1)
#         x = self.fc(x)

#         return x


# 
class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=10, pretrained=False,
                 scheduler_type='none', total_epochs=100, final_skip_values=None,
                 update_per_batch=False, start_value=1.0, min_bitwidth=32,
                 enable_quantization=False):

        super(ResNet, self).__init__()
        self.in_channels = 64
        self.block = block
        self.scheduler_type = scheduler_type
        self.total_epochs = total_epochs
        self.final_skip_values = final_skip_values or [1.0] * 4
        self.update_per_batch = update_per_batch
        self.start_value = start_value
        self.min_bitwidth = min_bitwidth
        self.enable_quantization = enable_quantization

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0], stride=1, stage=0, final_skip=self.final_skip_values[0], start_value=self.start_value, min_bitwidth=self.min_bitwidth, enable_quantization=self.enable_quantization)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, stage=1, final_skip=self.final_skip_values[1], start_value=self.start_value, min_bitwidth=self.min_bitwidth, enable_quantization=self.enable_quantization)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, stage=2, final_skip=self.final_skip_values[2], start_value=self.start_value, min_bitwidth=self.min_bitwidth, enable_quantization=self.enable_quantization)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, stage=3, final_skip=self.final_skip_values[3], start_value=self.start_value, min_bitwidth=self.min_bitwidth, enable_quantization=self.enable_quantization)


        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        self._initialize_weights()

    def _make_layer(self, block, out_channels, blocks, stride=1, stage=None,
                final_skip=0.0, start_value=1.0, min_bitwidth=32, enable_quantization=False):
        
        
        expansion = getattr(block, 'expansion', 1)
        layers = []
        block_sig = signature(block)
        

        for i in range(blocks):
            
            in_c = self.in_channels
            out_c = out_channels
            current_stride = stride if i == 0 else 1

            
            # ✅ Only set downsample for the FIRST block
            if i == 0 and (current_stride != 1 or in_c != out_c * expansion):
                downsample = nn.Sequential(
                    nn.Conv2d(in_c, out_c * expansion, kernel_size=1, stride=current_stride, bias=False),
                    nn.BatchNorm2d(out_c * expansion),
                )
            else:
                downsample = None  # clear it for later blocks

            block_args = {
                'stride': current_stride,
            }

            if 'downsample' in block_sig.parameters:
                block_args['downsample'] = downsample

            if 'scheduler_type' in block_sig.parameters:
                block_args.update({
                    'scheduler_type': self.scheduler_type,
                    'total_epochs': self.total_epochs,
                    'final_skip': final_skip,
                    'start_value': start_value
                })

            if 'min_bitwidth' in block_sig.parameters:
                block_args['min_bitwidth'] = min_bitwidth

            if 'enable_quantization' in block_sig.parameters:
                block_args['enable_quantization'] = enable_quantization

            layers.append(block(in_c, out_c, **block_args))
            self.in_channels = out_c * expansion  # ✅ Safe to update here
            

        return nn.Sequential(*layers)

    def update_skip_scale(self, step, total_steps):
        for layer in [self.layer1, self.layer2, self.layer3, self.layer4]:
            for block in layer:
                if hasattr(block, "update_skip_scale"):
                    block.update_skip_scale(step, total_steps)

    def get_skip_scales(self):
        return {
            f"stage{i+1}": getattr(self, f"layer{i+1}")[0].skip_scale
            for i in range(4)
            if hasattr(getattr(self, f"layer{i+1}")[0], 'skip_scale')
        }

    def get_quantization_bitwidths(self):
        return {
            f"stage{i+1}": getattr(self, f"layer{i+1}")[0].quantization_bitwidth
            for i in range(4)
            if hasattr(getattr(self, f"layer{i+1}")[0], 'quantization_bitwidth')
        }

    def _initialize_weights(self):
        for m in self.modules():

            if isinstance(m, DiracConv2d):
                continue  # skip: already initialized with Dirac + alpha/beta

            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


# Factory function
def resnet50(num_classes=10, pretrained=False, scheduler_type='linear', total_epochs=100, final_skip_values=None, update_per_batch=False, start_value=None, min_bitwidth=32, enable_quantization=False):
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes, pretrained=pretrained, scheduler_type=scheduler_type, total_epochs=total_epochs, final_skip_values=final_skip_values, update_per_batch=update_per_batch, start_value=start_value, min_bitwidth=min_bitwidth, enable_quantization=enable_quantization)

def resnet50bn(num_classes=10, pretrained=False, scheduler_type='linear', total_epochs=100, final_skip_values=None, update_per_batch=False):
    return ResNet(Bottleneck_bn, [3, 4, 6, 3], num_classes=num_classes, pretrained=pretrained, scheduler_type=scheduler_type, total_epochs=total_epochs, final_skip_values=final_skip_values, use_bn=True, update_per_batch=update_per_batch)

def resnet18(num_classes=10, pretrained=False, scheduler_type='linear', total_epochs=100, final_skip_values=None, update_per_batch=False, start_value=1.0,  min_bitwidth=32, enable_quantization=False):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes, pretrained=pretrained, scheduler_type=scheduler_type, total_epochs=total_epochs, final_skip_values=final_skip_values, update_per_batch=update_per_batch, start_value=start_value, min_bitwidth=min_bitwidth, enable_quantization=enable_quantization)

def resnet34(num_classes=10, pretrained=False, scheduler_type='linear', total_epochs=100, final_skip_values=None, update_per_batch=False, start_value=1.0, min_bitwidth=32, enable_quantization=False):
    
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes, pretrained=pretrained, scheduler_type=scheduler_type, total_epochs=total_epochs, final_skip_values=final_skip_values, update_per_batch=update_per_batch, start_value=start_value, min_bitwidth=min_bitwidth, enable_quantization=enable_quantization)

def resnet101(num_classes=10, pretrained=False, scheduler_type='linear', total_epochs=100, final_skip_values=None, update_per_batch=False, start_value=None, min_bitwidth=32, enable_quantization=False):
    return ResNet(Bottleneck, [3, 4, 23, 3],  
        num_classes=num_classes,
        pretrained=pretrained,
        scheduler_type=scheduler_type,
        total_epochs=total_epochs,
        final_skip_values=final_skip_values,
        update_per_batch=update_per_batch,
        start_value=start_value,
        min_bitwidth=min_bitwidth,
        enable_quantization=enable_quantization
    )

def resnet152(num_classes=10, pretrained=False, scheduler_type='linear', total_epochs=100,
              final_skip_values=None, update_per_batch=False, start_value=None,
              min_bitwidth=32, enable_quantization=False):
    return ResNet(
        Bottleneck,
        [3, 8, 36, 3],  
        num_classes=num_classes,
        pretrained=pretrained,
        scheduler_type=scheduler_type,
        total_epochs=total_epochs,
        final_skip_values=final_skip_values,
        update_per_batch=update_per_batch,
        start_value=start_value,
        min_bitwidth=min_bitwidth,
        enable_quantization=enable_quantization
    )

def dirac18(num_classes=10):
    return ResNet(
        DiracBasicBlock,       # <--- the block we just defined
        [2, 2, 2, 2],          # ResNet-18 style architecture
        num_classes=num_classes,
        pretrained=False,
        scheduler_type='none',           # dummy
        total_epochs=1,                  # dummy
        final_skip_values=[0, 0, 0, 0],  # dummy
        update_per_batch=False,          # dummy
        start_value=0.0,                 # dummy
        min_bitwidth=32,                 # dummy
        enable_quantization=False        # dummy
    )

def dirac34(num_classes=10):
    return ResNet(
        DiracBasicBlock,       # <--- the block we just defined
        [3, 4, 6, 3],          # ResNet-18 style architecture
        num_classes=num_classes,
        pretrained=False,
        scheduler_type='none',           # dummy
        total_epochs=1,                  # dummy
        final_skip_values=[0, 0, 0, 0],  # dummy
        update_per_batch=False,          # dummy
        start_value=0.0,                 # dummy
        min_bitwidth=32,                 # dummy
        enable_quantization=False        # dummy
    )

def dirac50(num_classes=10):
    return ResNet(
        DiracBottleneck,
        [3, 4, 6, 3],  
        num_classes=num_classes,
        pretrained=False,
        scheduler_type='none',           # not used
        total_epochs=1,                  # dummy
        final_skip_values=[0, 0, 0, 0],  # dummy
        update_per_batch=False,          # dummy
        start_value=0.0,                 # dummy
        min_bitwidth=32,                 # dummy
        enable_quantization=False        # dummy
    )

    