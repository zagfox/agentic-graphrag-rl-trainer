from unsloth import FastLanguageModel
from transformers import PreTrainedModel, PreTrainedTokenizer
from omegaconf import DictConfig
import torch
from typing import Tuple


def load_model_and_tokenizer(
    cfg: DictConfig,
    hardware_cfg: DictConfig
) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    """
    Load model with Unsloth optimization for 2x speed, 70% less VRAM.

    Reference: https://github.com/unslothai/unsloth
    """
    # Get sequence length from config with fallback
    max_seq_length = cfg.model.get("max_sequence_length", 2048)

    # Validate sequence length based on available GPU memory
    if hardware_cfg.get("auto_tune_sequence_length", True):
        max_seq_length = _determine_optimal_sequence_length(hardware_cfg, max_seq_length)

        # CRITICAL: Update config with actual max_sequence_length used
        # This ensures dataset loader and GRPO trainer use the same limit
        cfg.model.max_sequence_length = max_seq_length
        if hasattr(cfg, 'dataset') and hasattr(cfg.dataset, 'preprocessing'):
            cfg.dataset.preprocessing.max_length = max_seq_length

    # Get dtype
    dtype = None  # Auto-detect
    if hardware_cfg.mixed_precision == "bf16":
        dtype = torch.bfloat16
    elif hardware_cfg.mixed_precision == "fp16":
        dtype = torch.float16

    # Load with Unsloth
    # Unsloth automatically handles quantization parameters when load_in_4bit=True
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=cfg.model.name_or_path,
        max_seq_length=max_seq_length,
        dtype=dtype,
        load_in_4bit=cfg.model.quantization.load_in_4bit,
        # Unsloth automatically uses optimal 4-bit config (nf4, double_quant, etc.)
    )

    # Apply LoRA using Unsloth's optimized PEFT
    model = FastLanguageModel.get_peft_model(
        model,
        r=cfg.model.lora.r,
        lora_alpha=cfg.model.lora.lora_alpha,
        lora_dropout=cfg.model.lora.lora_dropout,
        target_modules=cfg.model.lora.target_modules,
        bias=cfg.model.lora.bias,
        use_gradient_checkpointing="unsloth",  # Unsloth's optimized checkpointing
        random_state=42,
        use_rslora=cfg.model.lora.use_rslora,
    )

    # Setup tokenizer
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # For batch generation in RL

    # Enable gradient checkpointing if configured
    if hardware_cfg.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    return model, tokenizer


def prepare_model_for_rl(model: PreTrainedModel) -> PreTrainedModel:
    """
    Prepare model for RL training.

    For PPO, we need:
    - Model outputs logits (already the case)
    - Model can be used for generation
    - Value head will be added by TRL automatically
    """
    # Ensure model is in training mode
    model.train()

    # Enable gradient computation for LoRA parameters only
    for name, param in model.named_parameters():
        if "lora" in name.lower():
            param.requires_grad = True
        else:
            param.requires_grad = False

    return model


def _determine_optimal_sequence_length(hardware_cfg: DictConfig, requested_length: int) -> int:
    """Determine optimal sequence length based on GPU memory constraints."""
    if not torch.cuda.is_available():
        return min(requested_length, 512)  # Conservative for CPU

    total_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)

    # Memory mapping (approximate, needs tuning)
    # These are conservative estimates for Qwen3-8B with 4-bit quantization
    memory_to_sequence_map = {
        8: 512,    # 8GB cards
        12: 2048,  # 12GB cards (RTX 4070)
        16: 4096,  # 16GB cards (RTX 4080)
        24: 8192,  # 24GB cards (RTX 4090)
        32: 16384, # 32GB cards (A100)
        48: 32768, # 48GB cards (H100)
    }

    # Find closest memory tier
    optimal_length = 256  # Very conservative fallback
    for memory_threshold, max_sequence in sorted(memory_to_sequence_map.items()):
        if total_memory_gb >= memory_threshold:
            optimal_length = max_sequence

    # Use requested length if within limits
    final_length = min(requested_length, optimal_length)

    if final_length < requested_length:
        print(f"Warning: Reduced sequence length from {requested_length} to {final_length} due to GPU memory constraints")
        print(f"GPU Memory: {total_memory_gb:.1f}GB, Recommended max: {optimal_length}")

    return final_length
