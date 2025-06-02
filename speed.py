import torch
import time
import statistics
import yaml
from models import resnet50, resnet101  # Add any additional models here

torch.backends.cudnn.benchmark = True  # Enables autotuning for best performance

def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def get_model_by_name(name):
    model_map = {
        "resnet50": resnet50,
        "resnet101": resnet101,
    }
    if name not in model_map:
        raise ValueError(f"Unknown model name: {name}")
    return model_map[name](num_classes=10)

def measure_throughput(model, batch_size=1, num_batches=100, warmup=30, device="cuda"):
    model.eval().to(device)
    input_tensor = torch.randn(batch_size, 3, 224, 224, device=device)

    # Warm-up
    for _ in range(warmup):
        with torch.no_grad():
            _ = model(input_tensor)
    torch.cuda.synchronize()

    # Timed run
    start = time.time()
    with torch.no_grad():
        for _ in range(num_batches):
            _ = model(input_tensor)
    torch.cuda.synchronize()
    end = time.time()

    elapsed = end - start
    throughput = (batch_size * num_batches) / elapsed
    latency_per_batch = elapsed / num_batches
    latency_per_image = latency_per_batch / batch_size

    return throughput, latency_per_batch, latency_per_image

def benchmark_model(name, model, config):
    print(f"\n📦 Benchmarking: {name}")
    results = []
    latencies_batch = []
    latencies_image = []

    for i in range(config["repeats"]):
        throughput, latency_b, latency_i = measure_throughput(
            model,
            batch_size=config["batch_size"],
            num_batches=config["num_batches"],
            warmup=config["warmup"],
            device=config["device"]
        )
        results.append(throughput)
        latencies_batch.append(latency_b)
        latencies_image.append(latency_i)
        print(f"  Run {i+1}: {throughput:.2f} samples/sec | "
              f"{latency_b*1000:.2f} ms/batch | {latency_i*1000:.2f} ms/image")

    print(f"\n📊 {name} Summary (batch_size={config['batch_size']}):")
    print(f"  Throughput (samples/sec):")
    print(f"    Min:    {min(results):.2f}")
    print(f"    Max:    {max(results):.2f}")
    print(f"    Mean:   {statistics.mean(results):.2f}")
    print(f"    Median: {statistics.median(results):.2f}")
    print(f"  Latency per batch (ms):")
    print(f"    Min:    {min(latencies_batch)*1000:.2f}")
    print(f"    Max:    {max(latencies_batch)*1000:.2f}")
    print(f"    Mean:   {statistics.mean(latencies_batch)*1000:.2f}")
    print(f"    Median: {statistics.median(latencies_batch)*1000:.2f}")
    print(f"  Latency per image (ms):")
    print(f"    Min:    {min(latencies_image)*1000:.2f}")
    print(f"    Max:    {max(latencies_image)*1000:.2f}")
    print(f"    Mean:   {statistics.mean(latencies_image)*1000:.2f}")
    print(f"    Median: {statistics.median(latencies_image)*1000:.2f}")
    return results

def main():
    config = load_config("config.yaml")

    baseline_name = config.get("baseline_model")
    pruned_path = config.get("pruned_model_path")

    if baseline_name and not pruned_path:
        print(f"\n✅ Loading baseline model only: {baseline_name}")
        model = get_model_by_name(baseline_name)
        benchmark_model(f"Baseline {baseline_name}", model, config)

    elif pruned_path and not baseline_name:
        print(f"\n🔍 Loading only pruned model from: {pruned_path}")
        model = torch.load(pruned_path, map_location=config["device"])
        benchmark_model("Pruned Model", model, config)

    elif baseline_name and pruned_path:
        print(f"\n✅ Loading baseline model: {baseline_name}")
        baseline = get_model_by_name(baseline_name)
        benchmark_model(f"Baseline {baseline_name}", baseline, config)

        print(f"\n🔍 Loading pruned model from: {pruned_path}")
        pruned = torch.load(pruned_path, map_location=config["device"])
        benchmark_model("Pruned Model", pruned, config)

    else:
        raise ValueError("❌ At least one of 'baseline_model' or 'pruned_model_path' must be provided in config.yaml")

if __name__ == "__main__":
    main()
