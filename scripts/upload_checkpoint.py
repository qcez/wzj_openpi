#!/usr/bin/env python3
"""Upload checkpoint to HuggingFace Hub."""

import argparse
import logging
import os
import re
from pathlib import Path
from typing import Optional

from huggingface_hub import HfApi, login, Repository


def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )


def extract_step_from_path(checkpoint_dir: Path) -> Optional[str]:
    """Extract step number from checkpoint path.
    
    Examples:
        ./checkpoints/pi0_base_aloha_robotwin_lora/beat_block_hammer_jax/16000 -> 16000
        /path/to/5000 -> 5000
    
    Args:
        checkpoint_dir: Path to checkpoint directory
    
    Returns:
        Step number as string, or None if not found
    """
    # Try to extract step number from the last path component
    path_str = str(checkpoint_dir).rstrip("/")
    match = re.search(r'/(\d+)/?$', path_str)
    if match:
        return match.group(1)
    return None


def login_to_huggingface(token: Optional[str] = None):
    """Login to HuggingFace Hub.
    
    Args:
        token: HuggingFace token. If None, will prompt for token.
    """
    logging.info("Logging in to HuggingFace Hub...")
    if token:
        login(token=token, add_to_git_credential=True)
    else:
        login(add_to_git_credential=True)
    logging.info("✅ Successfully logged in to HuggingFace Hub")


def upload_checkpoint(
    checkpoint_dir: Path,
    repo_id: str,
    step : str,
    commit_message: Optional[str] = None,
    private: bool = False,
    
):
    """Upload checkpoint to HuggingFace Hub.
    
    Files will be organized by step number in subdirectories.
    For example, checkpoint at ./path/16000 will upload to repo_id/16000/
    
    Args:
        checkpoint_dir: Path to checkpoint directory
        repo_id: Repository ID in format 'username/repo_name'
        commit_message: Commit message for the upload
        private: Whether the repository should be private
    """
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")
    
    if not checkpoint_dir.is_dir():
        raise ValueError(f"Checkpoint path is not a directory: {checkpoint_dir}")
    
    logging.info(f"Uploading checkpoint from: {checkpoint_dir}")
    logging.info(f"Target repository: {repo_id}")
    logging.info(f"Private: {private}")
    
    # Extract step number from path
    # If step is 'none', don't extract step number
    if step and step.lower() == 'none':
        step_num = None
    elif step:
        # Use provided step value
        step_num = step
    else:
        # Auto-extract step from path
        step_num = extract_step_from_path(checkpoint_dir)
    
    if step_num:
        logging.info(f"Using step number: {step_num}")
    else:
        logging.info("Not organizing by step number")
    
    # Initialize API
    api = HfApi()
    
    # Create repo if it doesn't exist
    try:
        logging.info(f"Creating/accessing repository: {repo_id}")
        api.create_repo(
            repo_id=repo_id,
            repo_type="model",
            private=private,
            exist_ok=True,
        )
    except Exception as e:
        logging.error(f"Failed to create repository: {e}")
        raise
    
    # Upload entire folder at once (avoids rate limit issues)
    logging.info("Uploading checkpoint files as a single batch...")
    
    # Determine path in repo
    if step_num:
        path_in_repo = step_num
    else:
        path_in_repo = "/"
    
    try:
        commit_info = api.upload_folder(
            folder_path=str(checkpoint_dir),
            repo_id=repo_id,
            repo_type="model",
            path_in_repo=path_in_repo,
            commit_message=commit_message or f"Upload checkpoint: step {step_num or 'unknown'}",
            ignore_patterns=["*.pyc", "__pycache__"],  # Ignore common cache files
        )
        
        # Count uploaded files
        uploaded_count = len(list(checkpoint_dir.rglob("*"))) - len(list(checkpoint_dir.rglob("__pycache__*")))
        logging.info(f"✅ Successfully uploaded {uploaded_count} files to https://huggingface.co/{repo_id}")
        if step_num:
            logging.info(f"   Files organized under: {repo_id}/{step_num}/")
        logging.info(f"   Commit: {commit_info.commit_url}")
        
    except Exception as e:
        logging.error(f"Failed to upload folder: {e}")
        raise


def main():
    parser = argparse.ArgumentParser(
        description="Upload checkpoint to HuggingFace Hub",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Upload checkpoint with auto-detected step (extracts 16000 from path)
  python scripts/upload_checkpoint.py \
    --checkpoint-dir ./checkpoints/pi0_base_aloha_robotwin_lora/beat_block_hammer_jax/16000 \
    --repo-id qcez/beat-block-hammer-checkpoint
  # Files will be uploaded to: qcez/beat-block-hammer-checkpoint/16000/

  # Upload with explicit step number
  python scripts/upload_checkpoint.py \
    --checkpoint-dir ./checkpoints/pi0_base_aloha_robotwin_lora/beat_block_hammer_jax/my_checkpoint \
    --repo-id qcez/beat-block-hammer-checkpoint \
    --step 5000
  # Files will be uploaded to: qcez/beat-block-hammer-checkpoint/5000/

  # Upload without step-based organization (files in root)
  python scripts/upload_checkpoint.py \
    --checkpoint-dir ./checkpoints/pi0_base_aloha_robotwin_lora/beat_block_hammer_jax/16000 \
    --repo-id qcez/beat-block-hammer-checkpoint \
    --step none
  # Files will be uploaded to root of: qcez/beat-block-hammer-checkpoint/

  # Upload with private repo
  python scripts/upload_checkpoint.py \
    --checkpoint-dir ./checkpoints/pi0_base_aloha_robotwin_lora/beat_block_hammer_jax/5000 \
    --repo-id qcez/beat-block-hammer-checkpoint \
    --private
        """
    )
    
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        required=True,
        help="Path to checkpoint directory to upload"
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        required=True,
        help="HuggingFace repository ID (format: username/repo_name)"
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="HuggingFace API token (if not provided, will prompt)"
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Make the repository private"
    )
    parser.add_argument(
        "--commit-message",
        type=str,
        default=None,
        help="Custom commit message"
    )
    parser.add_argument(
        "--step",
        type=str,
        default=None,
        help="Step number for organizing files in repo subdirectory. Set to 'none' to disable step-based organization. If not provided, will auto-extract from checkpoint path (e.g., 16000 from .../16000)"
    )
    
    args = parser.parse_args()
    
    setup_logging()
    
    # Get token from environment if not provided
    token = args.token or os.environ.get("HF_TOKEN")
    
    try:
        # Login
        login_to_huggingface(token=token)
        
        # Upload
        upload_checkpoint(
            checkpoint_dir=args.checkpoint_dir,
            repo_id=args.repo_id,
            commit_message=args.commit_message,
            private=args.private,
            step=args.step,
        )
        
    except Exception as e:
        logging.error(f"❌ Upload failed: {e}")
        raise


if __name__ == "__main__":
    main()
