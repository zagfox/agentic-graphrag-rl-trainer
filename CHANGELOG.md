# Changelog

All notable changes to this project will be documented in this file.

---

## [v1.1.0] - 2025-11-01

### ✅ Fixed - Sequence Length Synchronization

**Critical Bug Fix:** Resolved sequence length mismatch issues between model, dataset, and GRPO trainer.

#### Issues Resolved:
1. **Sequence Length Mismatch**
   - Model auto-tuned to 512 tokens but dataset loaded 804-token sequences
   - GRPO trainer had hardcoded max_prompt_length
   - Training crashed at step 3-4

2. **Dataset Format Incompatibility**
   - GRPO expected different prompt format
   - Tokenization failed with type mismatch errors

#### Changes Made:

**Model Config Synchronization** (`src/models/model_loader.py`)
- Auto-tuned sequence length now propagates to entire config
- Dataset preprocessing uses synchronized max_length
- Ensures all components use same token limits

**Dynamic GRPO Sequence Allocation** (`src/training/grpo_trainer.py`)
- Accepts `model_config` parameter for max_sequence_length
- Dynamic 75/25 split: 75% prompt (tool schemas), 25% completion (JSON output)
- Example: 512 total → 384 prompt + 128 completion

**Dataset Format Simplification** (`src/data/dataset_loader.py`)
- Simplified to: `{prompt: List[Dict], ground_truth: List[Dict], query: str}`
- Removed conflicting "messages" field
- GRPO applies chat template internally

**Updated Components:**
- SFT warmstart formatting compatibility
- Negative samples handling
- Trainer initialization passes model_config

#### Results:
- ✅ Training stable beyond 44+ steps (previously crashed at step 3)
- ✅ Speed: ~15-17 seconds/step
- ✅ Memory: 2.6GB RAM, 7.2GB VRAM (RTX 4070)
- ✅ All JSON presets validated and working

#### Documentation:
- Added `SEQUENCE_LENGTH_FIX_SUMMARY.md` with technical details
- Updated `README.md` with latest status
- Updated `TRAINING_GUIDE.md` with auto-fix section
- Updated `configs/presets/README.md` with system status

---

## [v1.0.0] - 2025-10-XX

### Added - Initial Release

**Core Features:**
- GRPO-based RL training for Qwen3-8B
- QLoRA 4-bit quantization with Unsloth optimization
- JSON configuration system with 5 memory-optimized presets
- Composite reward function (syntax + type checking + semantic)
- Multi-dataset support (ToolBench, xLAM, TC-RAG)
- WandB integration for experiment tracking
- RTX 4070 optimizations (12GB VRAM target)

**Presets:**
- `ultra_low_memory_8gb.json` - 8GB RAM
- `low_memory_16gb.json` - 16GB RAM (recommended)
- `medium_memory_24gb.json` - 24GB RAM
- `high_memory_32gb.json` - 32GB+ RAM
- `quick_test.json` - 100 steps for testing

**Documentation:**
- Complete training guide with JSON configuration examples
- Hardware-specific preset selection guide
- Troubleshooting section
- Migration notes from PPO to GRPO

---

## Legend

- ✅ **Fixed** - Bug fixes
- 🎯 **Added** - New features
- 🔄 **Changed** - Changes in existing functionality
- ⚠️ **Deprecated** - Soon-to-be removed features
- 🗑️ **Removed** - Removed features
- 🔒 **Security** - Security fixes
