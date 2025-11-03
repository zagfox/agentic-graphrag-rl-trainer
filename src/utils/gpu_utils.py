import torch
from typing import Dict


def get_gpu_memory_stats() -> Dict[str, float]:
    """Get current GPU memory usage statistics."""
    if not torch.cuda.is_available():
        return {"error": "CUDA not available"}

    # Get GPU properties dynamically
    device = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device)
    total_memory_gb = props.total_memory / (1024**3)

    allocated = torch.cuda.memory_allocated(device) / (1024**3)
    reserved = torch.cuda.memory_reserved(device) / (1024**3)
    max_allocated = torch.cuda.max_memory_allocated(device) / (1024**3)

    free_memory_gb = total_memory_gb - allocated

    return {
        "device_id": device,
        "device_name": props.name,
        "total_memory_gb": round(total_memory_gb, 2),
        "allocated_gb": round(allocated, 2),
        "reserved_gb": round(reserved, 2),
        "max_allocated_gb": round(max_allocated, 2),
        "free_gb": round(free_memory_gb, 2),
        "utilization_percent": round((allocated / total_memory_gb) * 100, 1),
        "memory_efficiency": round((allocated / total_memory_gb) * 100, 1)
    }


def clear_gpu_memory():
    """Clear GPU cache to free memory."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def get_optimal_batch_size(model, max_sequence_length: int = 2048, target_memory_utilization: float = 0.8) -> int:
    """Automatically determine optimal batch size based on available GPU memory."""
    if not torch.cuda.is_available():
        return 1  # Fallback for CPU

    # Clear cache and measure baseline
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    baseline_memory = torch.cuda.memory_allocated() / (1024**3)
    total_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    available_memory = (total_memory * target_memory_utilization) - baseline_memory

    if available_memory <= 0:
        return 1

    # Test memory usage with different batch sizes
    test_batch_sizes = [1, 2, 4, 8, 16]
    optimal_batch_size = 1

    model.eval()
    with torch.no_grad():
        for batch_size in test_batch_sizes:
            try:
                # Create test input
                test_input = torch.randint(
                    low=1,
                    high=model.config.vocab_size if hasattr(model.config, 'vocab_size') else 50000,
                    size=(batch_size, max_sequence_length),
                    device=model.device
                )

                # Forward pass to measure memory
                _ = model(test_input)

                current_memory = torch.cuda.memory_allocated() / (1024**3)
                memory_used = current_memory - baseline_memory

                if memory_used <= available_memory:
                    optimal_batch_size = batch_size
                else:
                    break

                # Clear memory for next test
                torch.cuda.empty_cache()

            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    break
                else:
                    raise e

    return optimal_batch_size


def check_gpu_compatibility() -> Dict:
    """Check if GPU is compatible with the system requirements."""
    if not torch.cuda.is_available():
        return {
            "compatible": False,
            "reason": "CUDA not available",
            "fallback": "CPU training (very slow)"
        }

    device = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device)
    total_memory_gb = props.total_memory / (1024**3)

    # Check minimum requirements
    MIN_MEMORY_GB = 8.0
    RECOMMENDED_MEMORY_GB = 12.0

    compatibility = {
        "compatible": True,
        "device_name": props.name,
        "total_memory_gb": round(total_memory_gb, 2),
        "compute_capability": f"{props.major}.{props.minor}",
        "multiprocessor_count": props.multi_processor_count
    }

    # Add recommendations
    if total_memory_gb < MIN_MEMORY_GB:
        compatibility["compatible"] = False
        compatibility["reason"] = f"Insufficient GPU memory ({total_memory_gb:.1f}GB < {MIN_MEMORY_GB}GB)"
        compatibility["recommendations"] = [
            "Use smaller model (e.g., Qwen3-1.8B instead of 8B)",
            "Enable gradient checkpointing",
            "Use CPU training (not recommended)",
            "Upgrade GPU"
        ]
    elif total_memory_gb < RECOMMENDED_MEMORY_GB:
        compatibility["warnings"] = [
            f"GPU memory ({total_memory_gb:.1f}GB) is below recommended ({RECOMMENDED_MEMORY_GB}GB)",
            "Training may be slower with reduced batch sizes",
            "Consider using gradient checkpointing"
        ]

    # Check compute capability
    if props.major < 7:  # Older GPUs
        compatibility["warnings"] = compatibility.get("warnings", [])
        compatibility["warnings"].append(
            f"Compute capability {props.major}.{props.minor} may not support all optimizations"
        )

    return compatibility
