import os
import sys
import time
import json
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.custom_lane_net import ResLaneSegNet

def get_process_ram_mb():
    try:
        import psutil
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / (1024 * 1024)
    except Exception:
        return 0.0

def measure_inference_footprint(model_path: str = "checkpoints/best_model.pth", img_h: int = 384, img_w: int = 640):
    print("=" * 65)
    print("MEASURING RESLANESEGNET INFERENCE MEMORY FOOTPRINT & LATENCY")
    print("=" * 65)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_shape = (1, 3, img_h, img_w)

    # 1. Model File Size
    if os.path.exists(model_path):
        model_size_mb = os.path.getsize(model_path) / (1024 * 1024)
    else:
        model_size_mb = 0.0

    # 2. Model Parameters
    model = ResLaneSegNet(in_channels=3, num_classes=6)
    if os.path.exists(model_path):
        ckpt = torch.load(model_path, map_location="cpu")
        state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
        model.load_state_dict(state_dict)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # 3. Memory Footprint on Device
    ram_before = get_process_ram_mb()
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        vram_before = torch.cuda.memory_allocated() / (1024 * 1024)
    else:
        vram_before = 0.0

    model = model.to(device)
    model.eval()

    dummy_input = torch.randn(*input_shape, device=device)

    # Warmup
    for _ in range(10):
        with torch.no_grad():
            _ = model(dummy_input)

    if device.type == "cuda":
        torch.cuda.synchronize()

    # Measure Inference Latency & Peak Memory
    num_runs = 50
    start_time = time.time()
    for _ in range(num_runs):
        with torch.no_grad():
            _ = model(dummy_input)
        if device.type == "cuda":
            torch.cuda.synchronize()
    total_time = time.time() - start_time
    avg_latency_ms = (total_time / num_runs) * 1000.0
    fps = 1000.0 / avg_latency_ms

    ram_after = get_process_ram_mb()
    if device.type == "cuda":
        vram_allocated_mb = torch.cuda.memory_allocated() / (1024 * 1024)
        vram_peak_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
        vram_reserved_mb = torch.cuda.memory_reserved() / (1024 * 1024)
    else:
        vram_allocated_mb = 0.0
        vram_peak_mb = 0.0
        vram_reserved_mb = 0.0

    results = {
        "device": str(device),
        "device_name": torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU",
        "input_resolution": f"{img_w}x{img_h}",
        "model_file_size_mb": round(model_size_mb, 2),
        "total_parameters": total_params,
        "total_parameters_million": round(total_params / 1e6, 3),
        "system_ram_usage_mb": round(ram_after, 2),
        "gpu_vram_allocated_mb": round(vram_allocated_mb, 2),
        "gpu_vram_peak_mb": round(vram_peak_mb, 2),
        "gpu_vram_reserved_mb": round(vram_reserved_mb, 2),
        "avg_latency_ms": round(avg_latency_ms, 2),
        "throughput_fps": round(fps, 1)
    }

    print(f"Device                    : {results['device_name']} ({results['device']})")
    print(f"Input Resolution          : {results['input_resolution']}")
    print(f"Model File Size (.pth)    : {results['model_file_size_mb']} MB")
    print(f"Total Parameters          : {results['total_parameters']:,} ({results['total_parameters_million']}M)")
    print(f"Inference Latency         : {results['avg_latency_ms']} ms/frame")
    print(f"Inference Throughput      : {results['throughput_fps']} FPS")
    print(f"System RAM In-Use         : {results['system_ram_usage_mb']} MB")
    if device.type == "cuda":
        print(f"GPU VRAM Allocated        : {results['gpu_vram_allocated_mb']} MB")
        print(f"GPU VRAM Peak Footprint   : {results['gpu_vram_peak_mb']} MB")
        print(f"GPU VRAM Reserved Cache   : {results['gpu_vram_reserved_mb']} MB")
    print("=" * 65)

    with open("memory_footprint.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print("Memory footprint successfully saved to memory_footprint.json")

    return results

if __name__ == "__main__":
    measure_inference_footprint()
