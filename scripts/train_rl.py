import argparse
import json
import sys
from pathlib import Path

import hydra
from dotenv import load_dotenv
from omegaconf import DictConfig, OmegaConf

# Load environment variables from .env file (HF_TOKEN, WANDB_API_KEY)
load_dotenv()

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.dataset_loader import FunctionCallingDatasetLoader
from src.models.model_loader import load_model_and_tokenizer, prepare_model_for_rl
from src.reward.composite_reward import CompositeRewardFunction
from src.training.grpo_trainer import AgenticRLTrainer
from src.utils.gpu_utils import get_gpu_memory_stats
from src.utils.logging import setup_logging
from src.utils.reproducibility import set_seed


def load_json_overrides(json_path: Path) -> list:
    """
    Load JSON config file and convert to Hydra overrides.

    Args:
        json_path: Path to JSON configuration file

    Returns:
        List of Hydra override strings
    """
    with open(json_path, "r") as f:
        json_config = json.load(f)

    overrides = []

    def flatten_dict(d, parent_key=""):
        for k, v in d.items():
            # Skip comment fields
            if k.startswith("_"):
                continue

            new_key = f"{parent_key}.{k}" if parent_key else k

            if isinstance(v, dict):
                flatten_dict(v, new_key)
            else:
                # Convert Python values to Hydra format
                if isinstance(v, bool):
                    value_str = str(v).lower()
                elif isinstance(v, (list, tuple)):
                    value_str = f"[{','.join(str(x) for x in v)}]"
                else:
                    value_str = str(v)

                overrides.append(f"{new_key}={value_str}")

    flatten_dict(json_config)
    return overrides


@hydra.main(version_base=None, config_path="../configs", config_name="default")
def main(cfg: DictConfig):
    """
    Main RL training script.

    Usage:
        # Default training
        uv run scripts/train_rl.py

        # Override parameters
        uv run scripts/train_rl.py \
            experiment.name=exp_002 \
            training.ppo.learning_rate=2e-5 \
            dataset=xlam

        # Multi-run for hyperparameter search
        uv run scripts/train_rl.py -m \
            training.ppo.learning_rate=1e-5,2e-5,5e-5
    """

    # Disable struct mode to allow config overrides
    OmegaConf.set_struct(cfg, False)

    # Print configuration
    print("=" * 80)
    print("Configuration:")
    print("=" * 80)
    print(OmegaConf.to_yaml(cfg))
    print("=" * 80)

    # Setup
    set_seed(cfg.experiment.seed)
    logger = setup_logging(cfg)

    logger.info(f"GPU Memory Stats: {get_gpu_memory_stats()}")

    # Load dataset
    logger.info("Loading and preparing dataset...")
    dataset_loader = FunctionCallingDatasetLoader(cfg)
    train_dataset, val_dataset, test_dataset = dataset_loader.load_and_prepare()

    logger.info(
        f"Train: {len(train_dataset)} | Val: {len(val_dataset)} | Test: {len(test_dataset)}"
    )

    # Load model
    logger.info("Loading model and tokenizer with QLoRA + Unsloth...")
    model, tokenizer = load_model_and_tokenizer(cfg, cfg.hardware)
    model = prepare_model_for_rl(model)

    logger.info(f"Model loaded. GPU Memory: {get_gpu_memory_stats()}")

    # Initialize reward function
    logger.info("Initializing composite reward function...")
    reward_fn = CompositeRewardFunction(cfg)

    # Initialize trainer
    logger.info("Initializing GRPO trainer...")
    trainer = AgenticRLTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        reward_function=reward_fn,
        config=cfg.training,
        experiment_config=cfg.experiment,
        model_config=cfg.model,  # Pass model config with auto-tuned max_sequence_length
    )

    # Training
    logger.info("Starting RL training...")
    trainer.train()

    logger.info("Training complete!")

    # Save final model
    output_path = Path(cfg.experiment.output_dir) / "final_model"
    trainer.save_model(output_path)
    logger.info(f"Model saved to {output_path}")


if __name__ == "__main__":
    # Parse --config-file argument before Hydra processes args
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "-conf", "--config-file", type=Path, help="Path to JSON config file"
    )
    args, remaining_args = parser.parse_known_args()

    # If JSON config provided, load overrides and inject into sys.argv
    if args.config_file:
        print(f"📄 Loading configuration from: {args.config_file}")

        if not args.config_file.exists():
            print(f"❌ Config file not found: {args.config_file}")
            sys.exit(1)

        try:
            json_overrides = load_json_overrides(args.config_file)
            print(f"✅ Loaded {len(json_overrides)} parameter overrides from JSON")

            # Replace sys.argv with script name + JSON overrides + remaining CLI args
            sys.argv = [sys.argv[0]] + json_overrides + remaining_args
        except Exception as e:
            print(f"❌ Error loading JSON config: {e}")
            sys.exit(1)

    main()
