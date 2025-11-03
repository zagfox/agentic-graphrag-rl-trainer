#!/usr/bin/env python3
"""
Quick test script to verify environment variables are loaded correctly.
"""

import os
from dotenv import load_dotenv

# Load .env file
load_dotenv()

print("=" * 80)
print("Environment Variables Test")
print("=" * 80)

# Test Hugging Face Token
hf_token = os.getenv("HF_TOKEN")
if hf_token:
    print(f"✅ HF_TOKEN found: {hf_token[:10]}...{hf_token[-4:]}")
else:
    print("❌ HF_TOKEN not found in environment")

# Test WandB API Key
wandb_key = os.getenv("WANDB_API_KEY")
if wandb_key:
    print(f"✅ WANDB_API_KEY found: {wandb_key[:10]}...{wandb_key[-4:]}")
else:
    print("❌ WANDB_API_KEY not found in environment")

print("=" * 80)

# Test Hugging Face login
if hf_token:
    print("\nTesting Hugging Face authentication...")
    try:
        from huggingface_hub import login, whoami
        login(token=hf_token, add_to_git_credential=False)
        user_info = whoami()
        print(f"✅ Hugging Face login successful!")
        print(f"   User: {user_info.get('name', 'N/A')}")
        print(f"   Type: {user_info.get('type', 'N/A')}")
    except Exception as e:
        print(f"❌ Hugging Face login failed: {e}")

# Test WandB login
if wandb_key:
    print("\nTesting WandB authentication...")
    try:
        import wandb
        wandb.login(key=wandb_key, relogin=True, verify=True)
        print(f"✅ WandB login successful!")
        print(f"   Logged in to: {wandb.api.viewer()}")
    except Exception as e:
        print(f"❌ WandB login failed: {e}")

print("=" * 80)
print("\n✨ If both tests passed, you're ready to start training!")
print("\nTo train with WandB enabled:")
print("  uv run python scripts/train_rl.py experiment.logging.use_wandb=true")
print("\nTo train with WandB disabled (local only):")
print("  uv run python scripts/train_rl.py experiment.logging.use_wandb=false")
