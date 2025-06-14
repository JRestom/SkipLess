import torch
from models import resnet50, resnet50bn, vit_base, resnet18, resnet34, dirac18, dirac34, dirac50, resnet101, resnet152
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

def build_model_from_config(config, rank, device):

        model_name = config['training']['model']
        pretrained_path = config['training'].get('pretrained_weights', None)
        num_classes = config['training']['num_classes']
        scheduler_type = config['training'].get('scheduler_type', "none")
        final_skip_values = config['model_params'].get('final_skip_values', [1.0]*4)
        start_value = config['model_params'].get('start_value', 1.0)
        total_epochs = config['training'].get('epochs', 100)
        min_bitwidth = config['training'].get('min_bitwidth', 8)
        enable_quantization = config['training'].get('enable_quantization', False)
        skip_scalar = config['training'].get('skip_scalar', 1.0)

        # Map model names to constructor functions
        model_constructors = {
            "resnet18": resnet18,
            "resnet34": resnet34,
            "resnet50": resnet50,
            "resnet101": resnet101,
            "resnet152": resnet152,
            "dirac18": dirac18,
            "dirac34": dirac34,
            "dirac50": dirac50,
            "vit_base": vit_base,
        }

        if model_name not in model_constructors:
            raise ValueError(f"❌ Unsupported model: {model_name}")


        map_location = {'cuda:%d' % 0: f'cuda:{rank}'}

        # === Case 1: Full model (.pt)
        if pretrained_path and pretrained_path.endswith(".pt"):
            model = torch.load(pretrained_path, map_location=map_location)
            print(f"[Rank {rank}] ✅ Loaded full model from: {pretrained_path}")

        # === Case 2: State dict (.pth)
        elif pretrained_path and pretrained_path.endswith(".pth"):
            # Instantiate first
            if "resnet" in model_name:
                model = model_constructors[model_name](
                    num_classes=num_classes,
                    scheduler_type=scheduler_type,
                    total_epochs=total_epochs,
                    final_skip_values=final_skip_values,
                    start_value=start_value,
                    min_bitwidth=min_bitwidth,
                    enable_quantization=enable_quantization
                )
            elif "vit" in model_name:
                model = vit_base(
                    num_classes=num_classes,
                    img_size=32,
                    patch=8,
                    hidden=384,
                    num_layers=7,
                    head=8,
                    dropout=0,
                    is_cls_token=True,
                    skip_scalar=skip_scalar,
                    start_value=start_value,
                    scheduler_type=scheduler_type,
                    total_epochs=total_epochs,
                    final_skip=final_skip_values[0]
                )
            else:
                model = model_constructors[model_name](num_classes=num_classes)

            state_dict = torch.load(pretrained_path, map_location=map_location)
            model.load_state_dict(state_dict)
            print(f"[Rank {rank}] ✅ Loaded state_dict from: {pretrained_path}")

        # === Case 3: No pretrained weights
        else:
            if rank == 0:
                print(f"[Rank {rank}] ⚠️ No pretrained weights provided. Initializing model from scratch.")
                
            if "resnet" in model_name:
                model = model_constructors[model_name](
                    num_classes=num_classes,
                    scheduler_type=scheduler_type,
                    total_epochs=total_epochs,
                    final_skip_values=final_skip_values,
                    start_value=start_value,
                    min_bitwidth=min_bitwidth,
                    enable_quantization=enable_quantization
                )
            elif "vit" in model_name:
                model = vit_base(
                    num_classes=num_classes,
                    img_size=32,
                    patch=8,
                    hidden=384,
                    num_layers=7,
                    head=8,
                    dropout=0,
                    is_cls_token=True,
                    skip_scalar=skip_scalar,
                    start_value=start_value,
                    scheduler_type=scheduler_type,
                    total_epochs=total_epochs,
                    final_skip=final_skip_values[0]
                )
            else:
                model = model_constructors[model_name](num_classes=num_classes)

        model = model.to(device)
        model = DDP(model, device_ids=[rank])
        return model

# Function to compute gradient norms per stage
def compute_stage_grad_norms(model):
    stage_gradients = {"stage1": 0, "stage2": 0, "stage3": 0, "stage4": 0}
    stage_counts = {"stage1": 0, "stage2": 0, "stage3": 0, "stage4": 0}

    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = torch.norm(param.grad).item()
            if "layer1" in name:
                stage_gradients["stage1"] += grad_norm
                stage_counts["stage1"] += 1
            elif "layer2" in name:
                stage_gradients["stage2"] += grad_norm
                stage_counts["stage2"] += 1
            elif "layer3" in name:
                stage_gradients["stage3"] += grad_norm
                stage_counts["stage3"] += 1
            elif "layer4" in name:
                stage_gradients["stage4"] += grad_norm
                stage_counts["stage4"] += 1

    for stage in stage_gradients:
        if stage_counts[stage] > 0:
            stage_gradients[stage] /= stage_counts[stage]

    return stage_gradients