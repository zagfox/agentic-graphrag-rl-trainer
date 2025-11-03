import logging
import sys
from pathlib import Path
from typing import Optional
import wandb
from omegaconf import DictConfig, OmegaConf


def setup_logging(cfg: DictConfig, log_file: Optional[Path] = None) -> logging.Logger:
    """Setup logging with file and console handlers."""
    logger = logging.getLogger("agentic_rag")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_format = logging.Formatter(
        '[%(asctime)s] %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)

    # File handler
    if log_file is None:
        log_file = Path(cfg.experiment.output_dir) / "training.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)

    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_format = logging.Formatter(
        '[%(asctime)s] %(levelname)s [%(filename)s:%(lineno)d] - %(message)s'
    )
    file_handler.setFormatter(file_format)
    logger.addHandler(file_handler)

    # Weights & Biases
    if cfg.experiment.logging.use_wandb:
        wandb.init(
            project=cfg.experiment.logging.wandb_project,
            name=cfg.experiment.name,
            config=OmegaConf.to_container(cfg, resolve=True),
            tags=["agentic-rag", "rl", "function-calling"],
        )
        logger.info("Weights & Biases logging initialized")

    return logger
