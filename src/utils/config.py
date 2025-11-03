from dataclasses import dataclass
from typing import List, Optional
from omegaconf import MISSING


@dataclass
class LoRAConfig:
    r: int = 16
    lora_alpha: int = 32
    target_modules: List[str] = MISSING
    lora_dropout: float = 0.05
    bias: str = "none"
    task_type: str = "CAUSAL_LM"
    use_rslora: bool = True


@dataclass
class QuantizationConfig:
    load_in_4bit: bool = True
    bnb_4bit_compute_dtype: str = "bfloat16"
    bnb_4bit_use_double_quant: bool = True
    bnb_4bit_quant_type: str = "nf4"


@dataclass
class ModelConfig:
    name_or_path: str = MISSING
    tokenizer_name: str = MISSING
    trust_remote_code: bool = True
    quantization: QuantizationConfig = QuantizationConfig()
    lora: LoRAConfig = LoRAConfig()
