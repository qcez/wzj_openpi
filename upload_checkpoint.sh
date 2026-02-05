#!/bin/bash
# Upload checkpoint to HuggingFace Hub

set -e

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Default values
PRIVATE=false

# Parse arguments
if [ $# -lt 2 ]; then
    echo "Usage: bash upload_checkpoint.sh <checkpoint_dir> <repo_id> [--step STEP] [--private] [--token TOKEN] [--commit-message MESSAGE]"
    echo ""
    echo "Examples:"
    echo "  # Auto-extract step from path (16000 from .../16000)"
    echo "  bash upload_checkpoint.sh ./checkpoints/pi0_base_aloha_robotwin_lora/beat_block_hammer_jax/16000 qcez/beat-block-hammer-checkpoint"
    echo "  # Files will be uploaded to: qcez/beat-block-hammer-checkpoint/16000/"
    echo ""
    echo "  # Explicit step number"
    echo "  bash upload_checkpoint.sh ./checkpoints/path/my_checkpoint qcez/beat-block-hammer-checkpoint --step 5000"
    echo "  # Files will be uploaded to: qcez/beat-block-hammer-checkpoint/5000/"
    echo ""
    echo "  # Disable step organization (files in root)"
    echo "  bash upload_checkpoint.sh ./checkpoints/pi0_base_aloha_robotwin_lora/beat_block_hammer_jax/16000 qcez/beat-block-hammer-checkpoint --step none"
    exit 1
fi

CHECKPOINT_DIR="$1"
REPO_ID="$2"
shift 2

# Parse optional arguments
TOKEN=""
COMMIT_MESSAGE=""
STEP=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --step)
            STEP="$2"
            shift 2
            ;;
        --private)
            PRIVATE=true
            shift
            ;;
        --token)
            TOKEN="$2"
            shift 2
            ;;
        --commit-message)
            COMMIT_MESSAGE="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Validate checkpoint directory
if [ ! -d "$CHECKPOINT_DIR" ]; then
    echo -e "${RED}❌ Checkpoint directory not found: $CHECKPOINT_DIR${NC}"
    exit 1
fi

echo -e "${BLUE}📦 Uploading checkpoint to HuggingFace Hub${NC}"
echo -e "${BLUE}   Checkpoint: $CHECKPOINT_DIR${NC}"
echo -e "${BLUE}   Repository: $REPO_ID${NC}"
if [ -n "$STEP" ]; then
    echo -e "${BLUE}   Step: $STEP${NC}"
fi
echo -e "${BLUE}   Private: $PRIVATE${NC}"
echo ""

# Build Python command
PYTHON_CMD="uv run python scripts/upload_checkpoint.py --checkpoint-dir $CHECKPOINT_DIR --repo-id $REPO_ID"

if [ -n "$STEP" ]; then
    PYTHON_CMD="$PYTHON_CMD --step $STEP"
fi

if [ "$PRIVATE" = true ]; then
    PYTHON_CMD="$PYTHON_CMD --private"
fi

if [ -n "$TOKEN" ]; then
    PYTHON_CMD="$PYTHON_CMD --token $TOKEN"
fi

if [ -n "$COMMIT_MESSAGE" ]; then
    PYTHON_CMD="$PYTHON_CMD --commit-message \"$COMMIT_MESSAGE\""
fi

# Run upload
echo -e "${BLUE}Running: $PYTHON_CMD${NC}"
echo ""

if eval "$PYTHON_CMD"; then
    echo -e "${GREEN}✅ Upload successful!${NC}"
    echo -e "${GREEN}   Repository: https://huggingface.co/$REPO_ID${NC}"
else
    echo -e "${RED}❌ Upload failed!${NC}"
    exit 1
fi
