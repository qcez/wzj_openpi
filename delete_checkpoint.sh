#!/bin/bash

# Delete checkpoint folder or entire repository from HuggingFace Hub

if [ $# -lt 1 ]; then
    echo "Usage: bash delete_checkpoint.sh <repo_id> [folder_path] [options]"
    echo ""
    echo "Examples:"
    echo "  # Delete specific checkpoint folder"
    echo "  bash delete_checkpoint.sh qcez/beat-block-hammer-checkpoint 16000"
    echo ""
    echo "  # Preview before deletion (dry run)"
    echo "  bash delete_checkpoint.sh qcez/beat-block-hammer-checkpoint 16000 --dry-run"
    echo ""
    echo "  # Delete dataset folder"
    echo "  bash delete_checkpoint.sh qcez/beat-block-hammer-dataset train --repo-type dataset"
    echo ""
    echo "  # Delete entire checkpoint repository (irreversible!)"
    echo "  bash delete_checkpoint.sh qcez/beat-block-hammer-checkpoint --delete-entire-repo"
    echo ""
    echo "  # Delete entire dataset repository"
    echo "  bash delete_checkpoint.sh qcez/beat-block-hammer-dataset --delete-entire-repo --repo-type dataset"
    exit 1
fi

repo_id=$1
shift

# Check if next argument is --delete-entire-repo
if [ "$1" = "--delete-entire-repo" ]; then
    uv run scripts/delete_checkpoint.py \
        --repo-id "$repo_id" \
        --delete-entire-repo \
        "$@"
else
    # Otherwise, it should be folder_path
    if [ -z "$1" ]; then
        echo "Error: folder_path must be provided"
        exit 1
    fi
    
    folder_path=$1
    shift
    
    uv run scripts/delete_checkpoint.py \
        --repo-id "$repo_id" \
        --folder-path "$folder_path" \
        "$@"
fi
