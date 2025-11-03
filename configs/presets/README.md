# 📋 Configuration Presets

Pre-configured JSON files optimized for different hardware configurations.

> **✅ System Status:** All sequence length synchronization issues resolved. GRPO now dynamically adjusts prompt/completion split (75%/25%) based on auto-tuned max_sequence_length.

## 🎯 Quick Usage

```bash
# Using -conf flag (short form)
uv run python scripts/train_rl.py -conf configs/presets/low_memory_16gb.json

# Using --config-file flag (long form)
uv run python scripts/train_rl.py --config-file configs/presets/quick_test.json
```

---

## 📊 Available Presets

### 1. **ultra_low_memory_8gb.json**
**For: 8GB RAM + RTX 3060 (8GB VRAM)**

```bash
uv run python scripts/train_rl.py -conf configs/presets/ultra_low_memory_8gb.json
```

**Specifications:**
- RAM Usage: ~5-6 GB
- Training Time: ~24-30 hours (10,000 steps)
- Batch Size: 1
- Num Generations: 1
- Sequence Length: 512 tokens
- SFT Warmstart: Disabled

**When to use:**
- Very limited RAM (<12GB)
- Older GPU with 8GB VRAM
- Testing on minimal hardware

---

### 2. **low_memory_16gb.json** ⭐ RECOMMENDED for RTX 4070
**For: 16GB RAM + RTX 4070 (12GB VRAM)**

```bash
uv run python scripts/train_rl.py -conf configs/presets/low_memory_16gb.json
```

**Specifications:**
- RAM Usage: ~8-10 GB
- Training Time: ~15-20 hours (10,000 steps)
- Batch Size: 2
- Num Generations: 2
- Sequence Length: 1024 tokens
- SFT Warmstart: Enabled (2 epochs)

**When to use:**
- Standard gaming PC with 16GB RAM
- RTX 4070 or similar 12GB GPU
- **Best balanced config for most users**

---

### 3. **medium_memory_24gb.json**
**For: 24GB RAM + RTX 4070/4080 (12-16GB VRAM)**

```bash
uv run python scripts/train_rl.py -conf configs/presets/medium_memory_24gb.json
```

**Specifications:**
- RAM Usage: ~12-16 GB
- Training Time: ~12-16 hours (10,000 steps)
- Batch Size: 3
- Num Generations: 3
- Sequence Length: 1536 tokens
- SFT Warmstart: Enabled (3 epochs)

**When to use:**
- Workstation with 24GB RAM
- RTX 4070/4080 GPU
- Want faster training with better quality

---

### 4. **high_memory_32gb.json**
**For: 32GB+ RAM + RTX 4080/4090 (16-24GB VRAM)**

```bash
uv run python scripts/train_rl.py -conf configs/presets/high_memory_32gb.json
```

**Specifications:**
- RAM Usage: ~16-22 GB
- Training Time: ~8-12 hours (10,000 steps)
- Batch Size: 4
- Num Generations: 4
- Sequence Length: 2048 tokens (full)
- SFT Warmstart: Enabled (3 epochs)

**When to use:**
- High-end workstation or server
- RTX 4090 or similar 24GB GPU
- Maximum performance and quality

---

### 5. **quick_test.json**
**For: Quick testing (any hardware)**

```bash
uv run python scripts/train_rl.py -conf configs/presets/quick_test.json
```

**Specifications:**
- RAM Usage: ~4-6 GB
- Training Time: ~30-60 minutes (100 steps)
- Batch Size: 2
- Num Generations: 2
- Sequence Length: 512 tokens
- SFT Warmstart: Disabled
- Total Steps: 100 (testing only)

**When to use:**
- Testing the pipeline before full training
- Debugging configurations
- Verifying setup works correctly

---

## 🔧 Customizing Presets

You can override any JSON parameter with command line arguments:

```bash
# Use preset + override specific parameters
uv run python scripts/train_rl.py \
    -conf configs/presets/low_memory_16gb.json \
    experiment.name=my_custom_run \
    training.schedule.total_steps=5000
```

---

## 📝 Creating Custom Presets

Copy any preset and modify it:

```bash
cp configs/presets/low_memory_16gb.json configs/presets/my_custom.json
# Edit my_custom.json with your parameters
uv run python scripts/train_rl.py -conf configs/presets/my_custom.json
```

---

## 🎛️ Key Parameters Explained

### Critical Memory Parameters

| Parameter | Impact | Values |
|-----------|--------|--------|
| `training.ppo.num_generations` | **Highest impact** | 1-4 (lower = less RAM) |
| `dataset.preprocessing.max_length` | High impact | 512-2048 (shorter = less RAM) |
| `training.ppo.batch_size` | Medium impact | 1-4 (smaller = less RAM) |
| `training.sft_warmstart.enabled` | Medium impact | true/false (false saves ~2-3GB) |

### Performance Parameters

| Parameter | Impact |
|-----------|--------|
| `training.schedule.total_steps` | Total training duration |
| `training.ppo.gradient_accumulation_steps` | Effective batch size |
| `training.ppo.learning_rate` | Convergence speed |

---

## 💡 Tips

1. **Start with quick_test.json** to verify everything works
2. **Use the lowest memory preset** that works for your hardware
3. **Monitor RAM usage** during training: `watch -n 1 free -h`
4. **Monitor GPU** during training: `watch -n 1 nvidia-smi`
5. **If OOM occurs**, drop down to next lower preset

---

## 📊 Memory Comparison Table

| Preset | RAM | VRAM | Batch×Gens | Seq Len | Time |
|--------|-----|------|------------|---------|------|
| Ultra Low | 8GB | 8GB | 1×1 = 1 | 512 | ~30h |
| Low ⭐ | 16GB | 12GB | 2×2 = 4 | 1024 | ~18h |
| Medium | 24GB | 12-16GB | 3×3 = 9 | 1536 | ~14h |
| High | 32GB+ | 16-24GB | 4×4 = 16 | 2048 | ~10h |
| Quick Test | Any | Any | 2×2 = 4 | 512 | ~1h |

---

## ❓ Which Preset Should I Use?

Run this command to check your system:

```bash
# Check RAM
free -h | grep Mem

# Check GPU
nvidia-smi --query-gpu=name,memory.total --format=csv
```

**Decision Tree:**
- **8GB RAM** → `ultra_low_memory_8gb.json`
- **16GB RAM** → `low_memory_16gb.json` ⭐ (recommended for RTX 4070)
- **24GB RAM** → `medium_memory_24gb.json`
- **32GB+ RAM** → `high_memory_32gb.json`

**For testing first:** Always start with `quick_test.json`!
