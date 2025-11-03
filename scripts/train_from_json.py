#!/usr/bin/env python3
"""
Training script that accepts JSON configuration files.

Usage:
    uv run python scripts/train_from_json.py configs/presets/low_memory.json
    uv run python scripts/train_from_json.py --config my_config.json
"""

import sys
import json
import argparse
from pathlib import Path
import subprocess

def json_to_hydra_overrides(json_config: dict) -> list:
    """
    Convert JSON config to Hydra command line overrides.

    Example:
        {"training": {"ppo": {"batch_size": 2}}}
        -> ["training.ppo.batch_size=2"]
    """
    overrides = []

    def flatten_dict(d, parent_key=''):
        for k, v in d.items():
            new_key = f"{parent_key}.{k}" if parent_key else k

            if isinstance(v, dict):
                flatten_dict(v, new_key)
            else:
                # Convert Python values to string format
                if isinstance(v, bool):
                    value_str = str(v).lower()
                elif isinstance(v, (list, tuple)):
                    value_str = f"[{','.join(str(x) for x in v)}]"
                else:
                    value_str = str(v)

                overrides.append(f"{new_key}={value_str}")

    flatten_dict(json_config)
    return overrides

def main():
    parser = argparse.ArgumentParser(
        description="Train RL model using JSON configuration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use preset configuration
  uv run python scripts/train_from_json.py configs/presets/low_memory.json

  # Custom JSON config
  uv run python scripts/train_from_json.py --config my_config.json

  # With additional overrides
  uv run python scripts/train_from_json.py my_config.json --override experiment.name=test_run
        """
    )

    parser.add_argument(
        'config_file',
        type=Path,
        help='Path to JSON configuration file'
    )

    parser.add_argument(
        '--override',
        nargs='*',
        default=[],
        help='Additional Hydra overrides (e.g., experiment.name=test)'
    )

    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Print command without executing'
    )

    args = parser.parse_args()

    # Read JSON config
    if not args.config_file.exists():
        print(f"❌ Config file not found: {args.config_file}")
        sys.exit(1)

    print(f"📄 Loading config from: {args.config_file}")

    with open(args.config_file, 'r') as f:
        json_config = json.load(f)

    # Convert to Hydra overrides
    hydra_overrides = json_to_hydra_overrides(json_config)

    # Add user overrides
    hydra_overrides.extend(args.override)

    # Build command
    cmd = [
        'uv', 'run', 'python', 'scripts/train_rl.py'
    ] + hydra_overrides

    print("\n🚀 Training command:")
    print(" ".join(cmd))
    print()

    if args.dry_run:
        print("🔍 Dry run mode - command not executed")
        return

    # Execute training
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Training failed with exit code {e.returncode}")
        sys.exit(e.returncode)
    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted by user")
        sys.exit(130)

if __name__ == "__main__":
    main()
