#!/usr/bin/env python3
"""Delete checkpoint folder from HuggingFace Hub."""

import argparse
import logging
import os
from typing import Optional

from huggingface_hub import HfApi, login


def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )


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


def delete_checkpoint_folder(
    repo_id: str,
    folder_path: str = None,
    commit_message: Optional[str] = None,
    dry_run: bool = False,
    repo_type: str = "model",
    delete_entire_repo: bool = False,
):
    """Delete checkpoint folder or entire repository from HuggingFace Hub.
    
    Args:
        repo_id: Repository ID in format 'username/repo_name'
        folder_path: Path to folder in repo to delete (e.g., '16000' or 'checkpoint/16000')
                    Ignored if delete_entire_repo is True
        commit_message: Custom commit message
        dry_run: If True, only list files that would be deleted without actually deleting
        repo_type: Type of repository ('model' or 'dataset')
        delete_entire_repo: If True, delete the entire repository (not just a folder)
    """
    # Initialize API
    api = HfApi()
    
    logging.info(f"Target repository: {repo_id}")
    logging.info(f"Repository type: {repo_type}")
    
    # Delete entire repository
    if delete_entire_repo:
        logging.warning(f"⚠️  You are about to DELETE THE ENTIRE REPOSITORY: {repo_id}")
        
        if dry_run:
            logging.warning("⚠️  DRY RUN MODE - Repository will NOT be deleted")
            return
        
        # Confirm deletion
        response = input(f"\n⚠️  Are you absolutely sure? This action is IRREVERSIBLE! Type 'delete {repo_id}' to confirm: ")
        if response != f"delete {repo_id}":
            logging.info("Deletion cancelled")
            return
        
        try:
            logging.info(f"Deleting entire repository: {repo_id}...")
            api.delete_repo(
                repo_id=repo_id,
                repo_type=repo_type,
            )
            logging.info(f"✅ Successfully deleted entire repository: {repo_id}")
        except Exception as e:
            logging.error(f"Failed to delete repository: {e}")
            raise
        return
    
    # Delete specific folder
    if not folder_path:
        raise ValueError("folder_path must be provided if delete_entire_repo is False")
    
    logging.info(f"Folder to delete: {folder_path}")
    
    # Normalize folder path
    folder_path = folder_path.rstrip("/")
    
    if dry_run:
        logging.warning("⚠️  DRY RUN MODE - No files will actually be deleted")
    
    try:
        # List files in the folder
        logging.info("Fetching file list from repository...")
        file_list = api.list_repo_files(
            repo_id=repo_id,
            repo_type=repo_type,
        )
        
        # Filter files in the target folder
        files_to_delete = [
            f for f in file_list 
            if f.startswith(folder_path + "/") or f == folder_path
        ]
        
        if not files_to_delete:
            logging.warning(f"⚠️  No files found in folder: {folder_path}")
            return
        
        logging.info(f"Found {len(files_to_delete)} files to delete:")
        for file_path in files_to_delete:
            logging.info(f"  - {file_path}")
        
        if dry_run:
            logging.info("✅ DRY RUN completed (no files deleted)")
            return
        
        # Confirm deletion
        response = input(f"\n⚠️  Are you sure you want to delete {len(files_to_delete)} files? (yes/no): ")
        if response.lower() not in ["yes", "y"]:
            logging.info("Deletion cancelled")
            return
        
        # Delete files
        logging.info("Deleting files...")
        deleted_count = 0
        
        for file_path in files_to_delete:
            try:
                api.delete_file(
                    path_in_repo=file_path,
                    repo_id=repo_id,
                    repo_type=repo_type,
                    commit_message=commit_message or f"Delete checkpoint folder: {folder_path}",
                )
                logging.info(f"  ✓ Deleted: {file_path}")
                deleted_count += 1
            except Exception as e:
                logging.error(f"  ✗ Failed to delete {file_path}: {e}")
        
        logging.info(f"\n✅ Successfully deleted {deleted_count}/{len(files_to_delete)} files from {repo_id}/{folder_path}")
        
    except Exception as e:
        logging.error(f"Failed to delete folder: {e}")
        raise


def main():
    parser = argparse.ArgumentParser(
        description="Delete checkpoint folder or entire repository from HuggingFace Hub",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Delete specific checkpoint folder (model repo)
  python scripts/delete_checkpoint.py \
    --repo-id qcez/beat-block-hammer-checkpoint \
    --folder-path 16000

  # Preview files before deletion (dry run)
  python scripts/delete_checkpoint.py \
    --repo-id qcez/beat-block-hammer-checkpoint \
    --folder-path 16000 \
    --dry-run

  # Delete dataset folder
  python scripts/delete_checkpoint.py \
    --repo-id qcez/beat-block-hammer-dataset \
    --folder-path train \
    --repo-type dataset

  # Delete entire checkpoint repository (irreversible!)
  python scripts/delete_checkpoint.py \
    --repo-id qcez/beat-block-hammer-checkpoint \
    --delete-entire-repo

  # Delete entire dataset repository (irreversible!)
  python scripts/delete_checkpoint.py \
    --repo-id qcez/beat-block-hammer-dataset \
    --delete-entire-repo \
    --repo-type dataset

  # Delete nested checkpoint folder
  python scripts/delete_checkpoint.py \
    --repo-id qcez/fold-cloth-checkpoint \
    --folder-path checkpoints/fold_cloth_jax/5000
        """
    )
    
    parser.add_argument(
        "--repo-id",
        type=str,
        required=True,
        help="HuggingFace repository ID (format: username/repo_name)"
    )
    parser.add_argument(
        "--folder-path",
        type=str,
        default=None,
        help="Path to folder in repo to delete (e.g., '16000' or 'checkpoint/16000'). Required if --delete-entire-repo is not set"
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="HuggingFace API token (if not provided, will prompt)"
    )
    parser.add_argument(
        "--commit-message",
        type=str,
        default=None,
        help="Custom commit message"
    )
    parser.add_argument(
        "--repo-type",
        type=str,
        choices=["model", "dataset"],
        default="model",
        help="Type of repository: 'model' (checkpoint) or 'dataset'"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview files to be deleted without actually deleting them"
    )
    parser.add_argument(
        "--delete-entire-repo",
        action="store_true",
        help="Delete the entire repository (IRREVERSIBLE!). Requires explicit confirmation"
    )
    
    args = parser.parse_args()
    
    setup_logging()
    
    # Validate arguments
    if not args.delete_entire_repo and not args.folder_path:
        parser.error("Either --folder-path or --delete-entire-repo must be provided")
    
    # Get token from environment if not provided
    token = args.token or os.environ.get("HF_TOKEN")
    
    try:
        # Login
        login_to_huggingface(token=token)
        
        # Delete
        delete_checkpoint_folder(
            repo_id=args.repo_id,
            folder_path=args.folder_path,
            commit_message=args.commit_message,
            dry_run=args.dry_run,
            repo_type=args.repo_type,
            delete_entire_repo=args.delete_entire_repo,
        )
        
    except Exception as e:
        logging.error(f"❌ Delete failed: {e}")
        raise


if __name__ == "__main__":
    main()
