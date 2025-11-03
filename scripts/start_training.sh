#!/bin/bash
# ==============================================================================
# Agentic GraphRAG RL Training - Automated Workflow Script
# ==============================================================================
# This script handles the complete training workflow with error handling,
# automatic restarts, and comprehensive logging.
#
# Usage:
#   ./scripts/start_training.sh                    # Default training
#   ./scripts/start_training.sh --no-wandb         # Disable WandB
#   ./scripts/start_training.sh --experiment my_exp # Custom experiment name
#   ./scripts/start_training.sh --help             # Show help
# ==============================================================================

set -e  # Exit on error (disabled during training loop)

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default configuration
USE_WANDB=true
EXPERIMENT_NAME=""
LEARNING_RATE=""
BATCH_SIZE=""
DATASET=""
AUTO_RESTART=false
MAX_RESTARTS=3

# ==============================================================================
# Helper Functions
# ==============================================================================

print_header() {
    echo -e "${BLUE}=============================================================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}=============================================================================${NC}"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

show_help() {
    cat << EOF
Usage: ./scripts/start_training.sh [OPTIONS]

Automated training script for Agentic GraphRAG RL Trainer

OPTIONS:
    --wandb             Enable WandB logging (default: true)
    --no-wandb          Disable WandB logging
    --experiment NAME   Set custom experiment name
    --learning-rate LR  Set learning rate (e.g., 2e-5)
    --batch-size SIZE   Set batch size (e.g., 8)
    --dataset NAME      Set dataset (toolbench, xlam, tc_rag)
    --auto-restart      Enable automatic restart on crash (max 3 times)
    --help              Show this help message

EXAMPLES:
    # Default training with WandB
    ./scripts/start_training.sh

    # Custom experiment without WandB
    ./scripts/start_training.sh --no-wandb --experiment my_test

    # Hyperparameter tuning
    ./scripts/start_training.sh --learning-rate 2e-5 --batch-size 8

    # Different dataset with auto-restart
    ./scripts/start_training.sh --dataset xlam --auto-restart

EOF
    exit 0
}

# ==============================================================================
# Parse Command Line Arguments
# ==============================================================================

while [[ $# -gt 0 ]]; do
    case $1 in
        --wandb)
            USE_WANDB=true
            shift
            ;;
        --no-wandb)
            USE_WANDB=false
            shift
            ;;
        --experiment)
            EXPERIMENT_NAME="$2"
            shift 2
            ;;
        --learning-rate)
            LEARNING_RATE="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --dataset)
            DATASET="$2"
            shift 2
            ;;
        --auto-restart)
            AUTO_RESTART=true
            shift
            ;;
        --help)
            show_help
            ;;
        *)
            print_error "Unknown option: $1"
            show_help
            ;;
    esac
done

# ==============================================================================
# Pre-flight Checks
# ==============================================================================

print_header "PRE-FLIGHT CHECKS"

# Check if .env exists
if [ ! -f ".env" ]; then
    print_error ".env file not found!"
    echo "Please create .env with HF_TOKEN and WANDB_API_KEY"
    exit 1
fi
print_success ".env file found"

# Check GPU
if ! command -v nvidia-smi &> /dev/null; then
    print_error "nvidia-smi not found - CUDA not available"
    exit 1
fi

GPU_INFO=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -n1)
print_success "GPU detected: $GPU_INFO"

# Check Python environment
if ! command -v uv &> /dev/null; then
    print_error "uv not found - please install uv package manager"
    exit 1
fi
print_success "uv package manager found"

# Verify environment variables
print_info "Testing environment variables..."
if uv run python scripts/test_env.py > /tmp/env_test.log 2>&1; then
    print_success "Environment variables configured correctly"
else
    print_error "Environment variable test failed"
    cat /tmp/env_test.log
    exit 1
fi

# Check disk space
AVAILABLE_SPACE=$(df -BG . | tail -1 | awk '{print $4}' | sed 's/G//')
if [ "$AVAILABLE_SPACE" -lt 50 ]; then
    print_warning "Low disk space: ${AVAILABLE_SPACE}GB available (50GB+ recommended)"
else
    print_success "Disk space: ${AVAILABLE_SPACE}GB available"
fi

# ==============================================================================
# Build Training Command
# ==============================================================================

print_header "TRAINING CONFIGURATION"

TRAIN_CMD="uv run python scripts/train_rl.py"

# WandB
TRAIN_CMD="$TRAIN_CMD experiment.logging.use_wandb=$USE_WANDB"
print_info "WandB logging: $USE_WANDB"

# Experiment name
if [ -n "$EXPERIMENT_NAME" ]; then
    TRAIN_CMD="$TRAIN_CMD experiment.name=$EXPERIMENT_NAME"
    print_info "Experiment name: $EXPERIMENT_NAME"
fi

# Learning rate
if [ -n "$LEARNING_RATE" ]; then
    TRAIN_CMD="$TRAIN_CMD training.ppo.learning_rate=$LEARNING_RATE"
    print_info "Learning rate: $LEARNING_RATE"
fi

# Batch size
if [ -n "$BATCH_SIZE" ]; then
    TRAIN_CMD="$TRAIN_CMD training.ppo.batch_size=$BATCH_SIZE"
    print_info "Batch size: $BATCH_SIZE"
fi

# Dataset
if [ -n "$DATASET" ]; then
    TRAIN_CMD="$TRAIN_CMD dataset=$DATASET"
    print_info "Dataset: $DATASET"
fi

echo ""
print_info "Full training command:"
echo -e "${BLUE}$TRAIN_CMD${NC}"
echo ""

# ==============================================================================
# Training Loop with Auto-Restart
# ==============================================================================

print_header "STARTING TRAINING"

RESTART_COUNT=0
TRAINING_SUCCESS=false

if [ "$USE_WANDB" = true ]; then
    ENTITY=$(grep WANDB_ENTITY .env | cut -d'=' -f2)
    PROJECT="agentic-graphrag-rl-trainer"
    print_info "Monitor training at: https://wandb.ai/$ENTITY/$PROJECT"
fi

while [ "$RESTART_COUNT" -lt "$MAX_RESTARTS" ]; do
    if [ "$RESTART_COUNT" -gt 0 ]; then
        print_warning "Restart attempt $RESTART_COUNT of $MAX_RESTARTS"
        sleep 5
    fi

    print_info "Starting training... (Press Ctrl+C to stop)"
    echo ""

    # Run training
    if eval $TRAIN_CMD; then
        TRAINING_SUCCESS=true
        break
    else
        EXIT_CODE=$?
        print_error "Training failed with exit code $EXIT_CODE"

        if [ "$AUTO_RESTART" = false ]; then
            print_error "Auto-restart disabled. Exiting."
            exit $EXIT_CODE
        fi

        RESTART_COUNT=$((RESTART_COUNT + 1))

        if [ "$RESTART_COUNT" -ge "$MAX_RESTARTS" ]; then
            print_error "Maximum restart attempts reached. Exiting."
            exit $EXIT_CODE
        fi
    fi
done

# ==============================================================================
# Post-Training Summary
# ==============================================================================

if [ "$TRAINING_SUCCESS" = true ]; then
    print_header "TRAINING COMPLETED SUCCESSFULLY"

    # Find experiment directory
    if [ -n "$EXPERIMENT_NAME" ]; then
        EXP_DIR="experiments/$EXPERIMENT_NAME"
    else
        EXP_DIR="experiments/agentic_rag_baseline"
    fi

    if [ -d "$EXP_DIR" ]; then
        print_success "Training artifacts saved to: $EXP_DIR"

        # Count checkpoints
        CHECKPOINT_COUNT=$(find "$EXP_DIR" -type d -name "checkpoint-*" | wc -l)
        print_info "Checkpoints saved: $CHECKPOINT_COUNT"

        # Check final model
        if [ -d "$EXP_DIR/final_model" ]; then
            print_success "Final model saved: $EXP_DIR/final_model"
        fi

        # Training log
        if [ -f "$EXP_DIR/training.log" ]; then
            print_info "Training log: $EXP_DIR/training.log"
        fi
    fi

    echo ""
    print_header "NEXT STEPS"
    echo ""
    echo "1. Evaluate the model:"
    echo -e "   ${BLUE}uv run python scripts/evaluate.py model_path=$EXP_DIR/final_model${NC}"
    echo ""
    echo "2. View training logs:"
    echo -e "   ${BLUE}tail -f $EXP_DIR/training.log${NC}"
    echo ""
    if [ "$USE_WANDB" = true ]; then
        echo "3. View WandB dashboard:"
        echo -e "   ${BLUE}https://wandb.ai/$ENTITY/$PROJECT${NC}"
        echo ""
    fi
    echo "4. Compare checkpoints:"
    echo -e "   ${BLUE}ls -lh $EXP_DIR/checkpoint-*${NC}"
    echo ""

    print_success "All done! 🎉"
else
    print_error "Training failed after $RESTART_COUNT restart attempts"
    exit 1
fi
