import torch

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