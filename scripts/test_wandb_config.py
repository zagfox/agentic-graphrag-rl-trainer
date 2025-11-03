#!/usr/bin/env python3
"""
Test WandB configuration from Hydra config.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

import hydra
from omegaconf import DictConfig, OmegaConf
import os

@hydra.main(version_base=None, config_path="../configs", config_name="default")
def test_config(cfg: DictConfig):
    """Test WandB configuration."""
    print("=" * 80)
    print("WandB Configuration Test")
    print("=" * 80)

    # Print relevant config
    print("\n📋 Experiment Config:")
    print(f"  - Experiment name: {cfg.experiment.name}")
    print(f"  - WandB enabled: {cfg.experiment.logging.use_wandb}")
    print(f"  - WandB project: {cfg.experiment.logging.wandb_project}")

    print("\n🔑 Environment Variables:")
    print(f"  - WANDB_API_KEY: {'✅ Set' if os.getenv('WANDB_API_KEY') else '❌ Not set'}")
    print(f"  - WANDB_ENTITY: {os.getenv('WANDB_ENTITY', 'Not set (will use default)')}")
    print(f"  - HF_TOKEN: {'✅ Set' if os.getenv('HF_TOKEN') else '❌ Not set'}")

    print("\n🌐 WandB URL (when training starts):")
    entity = os.getenv('WANDB_ENTITY', 'your-username')
    project = cfg.experiment.logging.wandb_project
    print(f"  https://wandb.ai/{entity}/{project}")

    print("\n" + "=" * 80)
    print("✨ Configuration is ready!")
    print("\nTo start training with this config:")
    print("  uv run python scripts/train_rl.py experiment.logging.use_wandb=true")
    print("=" * 80)

if __name__ == "__main__":
    test_config()
