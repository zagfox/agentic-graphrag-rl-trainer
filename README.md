# Agentic GraphRAG RL Trainer

Reinforcement Learning system for training **Qwen3-8B** on function calling with AST-based reward functions, optimized for **RTX 4070 (12GB VRAM)**.

> **🚀 NEW USER?** Start here: [docs/guides/START_HERE.md](docs/guides/START_HERE.md) for immediate training!
>
> **✅ LATEST UPDATE (Nov 2025):** All sequence length synchronization issues resolved! Training is now fully stable. See [docs/development/SEQUENCE_LENGTH_FIX_SUMMARY.md](docs/development/SEQUENCE_LENGTH_FIX_SUMMARY.md) for technical details.

## 🔑 Key Features

- **Model**: Qwen3-8B (8.2B params) with QLoRA (4-bit quantization)
- **Optimization**: Unsloth for 2x speed, 70% less VRAM
- **RL Algorithm**: GRPO (Group Relative Policy Optimization) via TRL
- **Reward Functions**:
  - AST-based syntax validation
  - JSON schema type checking
  - BFCL semantic correctness (AST Score)
- **Datasets**: xLAM (Salesforce 60K) with synthetic dataset fallback
- **Memory**: ~11GB VRAM usage on RTX 4070

> **Note**: Migrated from PPO to GRPO for better performance and simpler API. See [docs/reference/migration-notes.md](docs/reference/migration-notes.md) for details.

## ⚡ Prerequisites

- Python 3.10+
- CUDA 12.1+
- NVIDIA RTX 4070 (or equivalent with 12GB+ VRAM)
- 32GB+ RAM recommended

## 🚀 Installation

### Quick Setup (Recommended - with uv)

```bash
# 1. Clone repository
git clone <your-repo-url>
cd agentic-graphrag-rl-trainer

# 2. Create environment with uv (recommended package manager)
uv venv
source .venv/bin/activate  # Linux/Mac

# 3. Install all dependencies
uv sync

# 4. Create .env file with your tokens
# HF_TOKEN=your_huggingface_token
# WANDB_API_KEY=your_wandb_key

# 5. Verify setup
uv run python scripts/test_env.py
```

### Alternative: Traditional pip installation

```bash
# 1. Clone repository
git clone <your-repo-url>
cd agentic-graphrag-rl-trainer

# 2. Create environment
uv venv
source .venv/bin/activate  # Linux/Mac

# 3. Install PyTorch with CUDA
uv add torch>=2.1.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 4. Install project
uv pip install -e .

# 5. Verify GPU
uv run python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

📚 **For detailed setup instructions**, see [docs/guides/QUICK_START.md](docs/guides/QUICK_START.md)

## 📖 Usage

### Quick Start - Automated Script (Recommended)

```bash
# One-command training with automatic pre-flight checks
./scripts/start_training.sh
```

This automated script:
- ✅ Verifies GPU and dependencies
- ✅ Tests environment variables
- ✅ Runs SFT warm-start (3 epochs)
- ✅ Trains with GRPO (10,000 steps)
- ✅ Saves checkpoints every 500 steps
- ✅ Logs to WandB dashboard
- ⏱️ **Takes 12-18 hours on RTX 4070**

**Options:**
```bash
# Without WandB
./scripts/start_training.sh --no-wandb

# Custom experiment name
./scripts/start_training.sh --experiment my_run

# Different hyperparameters
./scripts/start_training.sh --learning-rate 2e-5 --batch-size 8

# Help
./scripts/start_training.sh --help
```

### Alternative - Direct Python Script

```bash
# Basic training
uv run python scripts/train_rl.py

# With WandB enabled
uv run python scripts/train_rl.py experiment.logging.use_wandb=true
```

### Custom Configuration

```bash
# Change experiment name
uv run python scripts/train_rl.py experiment.name=my_experiment

# Use different dataset (xlam is default, or use synthetic_function_calling)
uv run python scripts/train_rl.py dataset=synthetic_function_calling

# Adjust learning rate
uv run python scripts/train_rl.py training.ppo.learning_rate=2e-5

# Enable SFT warm-start (recommended)
uv run python scripts/train_rl.py training.sft_warmstart.enabled=true

# Combine multiple overrides
uv run python scripts/train_rl.py \
    experiment.name=exp_002 \
    training.ppo.learning_rate=2e-5 \
    training.ppo.batch_size=8 \
    dataset=synthetic_function_calling
```

### Hyperparameter Search (Multi-run)

```bash
# Test different learning rates
uv run python scripts/train_rl.py -m \
    training.ppo.learning_rate=1e-5,2e-5,5e-5

# Grid search
uv run python scripts/train_rl.py -m \
    training.ppo.learning_rate=1e-5,2e-5 \
    training.ppo.batch_size=4,8
```

## ⚙️ Configuration

All configurations are in `configs/`:

- `default.yaml` - Main config with experiment settings
- `model/qwen3_8b.yaml` - Model architecture and LoRA settings
- `dataset/default.yaml` - xLAM dataset (Salesforce 60K) configuration
- `dataset/synthetic_function_calling.yaml` - Synthetic dataset configuration
- `training/ppo.yaml` - GRPO hyperparameters (named ppo for backward compatibility)
- `reward/composite_reward.yaml` - Reward function weights

### Example: Modify Reward Weights

Edit `configs/reward/composite_reward.yaml`:

```yaml
reward:
  components:
    syntax:
      weight: 0.4  # Increase syntax importance
    type_checking:
      weight: 0.2
    semantic:
      weight: 0.4  # Decrease semantic weight
```

## 📊 Monitoring

### WandB (Recommended)

```bash
# Login to WandB
wandb login

# Enable in config
uv run python scripts/train_rl.py experiment.logging.use_wandb=true
```

### TensorBoard

```bash
# Launch TensorBoard
tensorboard --logdir experiments/
```

### Logs

Training logs are saved to:
- `experiments/<experiment_name>/training.log`
- `experiments/<experiment_name>/checkpoint-<step>/`

## 🎯 Expected Performance

After full training (10,000 steps):

| Metric | Target |
|--------|--------|
| AST Score | >0.85 |
| Syntax Accuracy | >95% |
| Function Name Accuracy | >90% |
| Exact Match | >70% |

## 📁 Project Structure

```
agentic-graphrag-rl-trainer/
├── configs/                # Hydra configurations
│   ├── default.yaml
│   ├── model/
│   ├── dataset/
│   ├── training/
│   └── reward/
├── src/
│   ├── data/              # Dataset loading & preprocessing
│   ├── models/            # Model loading with QLoRA
│   ├── reward/            # AST-based reward functions
│   ├── training/          # GRPO trainer
│   ├── evaluation/        # Metrics
│   └── utils/             # Logging, GPU utils, etc.
├── scripts/
│   └── train_rl.py        # Main training script
├── docs/                  # Organized documentation
│   ├── guides/           # User guides and tutorials
│   ├── reference/        # Technical reference docs
│   ├── reports/          # Analysis reports
│   └── development/      # Development notes
├── logs/                  # Log directory
└── experiments/           # Training outputs (generated during training)
```

## 🔧 Technical Details

### Memory Optimization (RTX 4070 - 12GB)

1. **QLoRA 4-bit**: Reduces model from ~16GB to ~4GB
2. **Unsloth**: Additional 70% VRAM reduction + 2x speed
3. **GRPO**: No separate value/reference models (unlike PPO)
4. **Gradient Checkpointing**: Trades compute for memory
5. **Batch Size**: 4 with gradient accumulation
6. **Flash Attention**: 2x memory efficiency

**VRAM Breakdown:**
- Model (4-bit): ~4GB
- Activations: ~2GB
- Optimizer (LoRA only): ~1GB
- Generation buffer: ~3GB (4 completions per prompt)
- KV cache: ~1GB
- **Total: ~11GB** ✅

### Reward Function Pipeline

1. **Syntax Validation** (30% weight)
   - AST parsing with `json.loads()` / `ast.literal_eval()`
   - Structure validation (list of dicts, required fields)
   - Penalty: -1.0 for malformed output

2. **Type Checking** (20% weight)
   - JSON Schema validation with automatic schema inference
   - Parameter type correctness
   - Schema caching for efficiency

3. **Semantic Correctness** (50% weight)
   - BFCL AST Score:
     - 40% function name matching
     - 30% parameter name matching
     - 30% parameter value matching
   - Multi-call support with Hungarian algorithm

## 📚 Documentation

### Getting Started
- **[docs/guides/START_HERE.md](docs/guides/START_HERE.md)** - Quick start for immediate training
- **[docs/guides/QUICK_START.md](docs/guides/QUICK_START.md)** - All commands and options
- **[docs/guides/training-guide.md](docs/guides/training-guide.md)** - Complete setup guide

### Technical Guides
- **[docs/guides/how-it-works.md](docs/guides/how-it-works.md)** - Comprehensive technical documentation (1700+ lines)
- **[docs/reference/migration-notes.md](docs/reference/migration-notes.md)** - PPO to GRPO migration details
- **[docs/development/complete-implementing.md](docs/development/complete-implementing.md)** - Implementation details

### Scripts
- `scripts/start_training.sh` - Automated training with pre-flight checks
- `scripts/train_rl.py` - Main training script
- `scripts/test_env.py` - Test environment setup
- `scripts/test_wandb_config.py` - Verify WandB configuration
- `scripts/validate_implementations.py` - Validate all features

## 📚 References

- [GRPO Paper](https://arxiv.org/abs/2402.03300) - Group Relative Policy Optimization
- [Unsloth GRPO Guide](https://docs.unsloth.ai/get-started/reinforcement-learning-rl-guide) - Official guide
- [Unsloth](https://github.com/unslothai/unsloth) - Fast training optimization
- [TRL](https://github.com/huggingface/trl) - Transformer RL library
- [BFCL](https://gorilla.cs.berkeley.edu/leaderboard.html) - Function calling leaderboard
- [xLAM Dataset](https://huggingface.co/datasets/Salesforce/xlam-function-calling-60k) - Function calling dataset (60K examples)

## 🔧 Troubleshooting

### CUDA Out of Memory

```bash
# Reduce batch size
uv run python scripts/train_rl.py training.ppo.batch_size=2

# Reduce sequence length
uv run python scripts/train_rl.py dataset.preprocessing.max_length=1024
```

### Slow Training

```bash
# Check GPU utilization
nvidia-smi -l 1

# Verify Unsloth is installed
uv run python -c "import unsloth; print('Unsloth OK')"
```

### Dataset Not Found

```bash
# Download xLAM manually (requires HF_TOKEN)
huggingface-cli download Salesforce/xlam-function-calling-60k --repo-type dataset

# Or use synthetic dataset (no download required)
uv run python scripts/train_rl.py dataset=synthetic_function_calling

# Or specify custom local path
uv run python scripts/train_rl.py dataset.path=/path/to/local/dataset
```

## 📄 License

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

## 🤝 Contributing

Contributions welcome! Please open an issue or PR.

## 📞 Contact

[Your Contact Info]