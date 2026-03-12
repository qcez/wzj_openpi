import argparse
import torch
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
import sys
sys.path.append("/project/peilab/wzj/RoboTwin/policy/openpi_test/src")
from openpi.qwenvl.value_function import ValueFunction
import matplotlib.pyplot as plt
import numpy as np
import os
import json
from tqdm import tqdm

# Reduce verbose transformer generation warnings that repeat per-step
os.environ.setdefault('TRANSFORMERS_VERBOSITY', 'error')
try:
    from transformers import logging as _tr_logging
    _tr_logging.set_verbosity_error()
except Exception:
    pass

def load_model_and_processor(model_name_or_path, attn_implementation=None):
    """Load model and processor from checkpoint. (Adapted from test.py)"""
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    from openpi.qwenvl.utils.value_tokenizer import ValueTokenizer

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_name_or_path,
        attn_implementation=attn_implementation,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    processor = AutoProcessor.from_pretrained(model_name_or_path)
    model.eval()
    model.config.use_cache = True
    value_tokenizer = ValueTokenizer(
        llm_path=model_name_or_path,  # Use same model path to ensure consistency
        bins=201,
        min_value=-1.0,
        max_value=0.0,
    )
    return model, processor, value_tokenizer

def plot_episode_values(values, episode_idx, output_dir):
    """Plot values for a single episode. (Inspired by qwen_eval.py)"""
    steps = np.arange(len(values))
    plt.figure(figsize=(12, 6))
    plt.plot(steps, values, 'b-', label='Predicted Value', linewidth=2, marker='o', markersize=4)
    plt.xlabel('Step', fontsize=12)
    plt.ylabel('Value', fontsize=12)
    plt.title(f'Episode {episode_idx} - Value Prediction', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save figure
    plot_path = os.path.join(output_dir, f'episode_{episode_idx}_values.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved plot for episode {episode_idx} to {plot_path}")
    
    # Additional output similar to qwen_eval.py
    print(f"[DEBUG] Episode {episode_idx} summary:")
    print(f"  Total steps: {len(values)}")
    print(f"  Value range: {min(values):.4f} to {max(values):.4f}")
    print(f"  Mean value: {np.mean(values):.4f}")
    print(f"  Std value: {np.std(values):.4f}")
    if len(values) >= 2:
        print(f"  First value: {values[0]:.4f}, Last value: {values[-1]:.4f}")
    print("=" * 80)

def main(args):
    print(f"Loading dataset from: {args.dataset_repo}")
    print("Loading ValueFunction model...")
    
    # Load dataset
    data = LeRobotDataset(repo_id=args.dataset_repo)
    dataset_meta = LeRobotDatasetMetadata(repo_id=args.dataset_repo)
    
    # Create a mock config object for ValueFunction
    class MockConfig:
        def __init__(self, model_path, repo_id):
            self.value_function_path = model_path
            class Data:
                def __init__(self, repo_id):
                    self.repo_id = repo_id
            self.data = Data(repo_id)
            
    config = MockConfig(args.model_path, args.dataset_repo)
    
    # Load ValueFunction
    value_function = ValueFunction(config)
    print(f"ValueFunction loaded from: {args.model_path}")
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Output directory: {args.output_dir}")
    
    num_all_episodes = len(dataset_meta.episodes)
    # Determine episode range to evaluate
    start_ep = args.start_episode if getattr(args, 'start_episode', None) is not None else 0
    end_ep = args.end_episode if getattr(args, 'end_episode', None) is not None else num_all_episodes - 1
    if start_ep < 0 or end_ep < start_ep or end_ep >= num_all_episodes:
        raise ValueError(f"Invalid episode range: start={start_ep}, end={end_ep}, available=0..{num_all_episodes-1}")
    selected_episodes = list(range(start_ep, end_ep + 1))
    print(f"Starting evaluation on episodes {start_ep}..{end_ep} (total {len(selected_episodes)})")
    
    # Collect all results
    all_results = {}
    
    # Compute values for each episode
    # Compute starting frame index for the first selected episode
    episode_start_idx = sum(dataset_meta.episodes[i]['length'] for i in range(start_ep))
    for episode_idx in tqdm(selected_episodes, desc="Evaluating episodes"):
        episode_length = dataset_meta.episodes[episode_idx]['length']
        values = []
        
        print(f"Processing episode {episode_idx} (length: {episode_length})...")
        
        for step in range(episode_length):
            frame_idx = episode_start_idx + step
            obs = data[frame_idx]
            
            # Prepare obs_batch: group cameras as required by ValueFunction
            # LeRobot observations are typically (C, H, W). We need to ensure we keep (C, H, W) 
            # so that ValueFunction's permute(2, 1, 0) results in (H, W, C).
            cam_high = obs['observation.images.cam_high']
            cam_left = obs['observation.images.cam_left_wrist']
            cam_right = obs['observation.images.cam_right_wrist']
            
            # Build obs_batch as a single-batch tensor wrapped in an object that
            # provides a `.device` attribute so `ValueFunction.predict_value`
            # (which expects a tensor-like batch) works without modifying it.
            # Build obs as a list of 3 (C,H,W) tensors on CPU so ToPILImage works
            # Convert floats in [0,1] to uint8 in [0,255] on CPU to satisfy ToPILImage
            def to_uint8_cpu(t):
                t_cpu = t.detach().cpu()
                if t_cpu.dtype.is_floating_point:
                    return (t_cpu * 255.0).clamp(0, 255).to(torch.uint8)
                return t_cpu.to(torch.uint8)

            cam_high_cpu = to_uint8_cpu(cam_high)
            cam_left_cpu = to_uint8_cpu(cam_left)
            cam_right_cpu = to_uint8_cpu(cam_right)

            class _BatchWrapper(list):
                def __init__(self, items, device):
                    super().__init__(items)
                    self.device = device

            # Use the model device so predict_value can read the device attribute
            try:
                model_device = next(value_function.model.parameters()).device
            except Exception:
                model_device = torch.device('cpu')

            # obs_batch is a list whose single element is a list of three (C,H,W) tensors
            obs_batch = _BatchWrapper([[cam_high_cpu, cam_left_cpu, cam_right_cpu]], model_device)
            # obs_batch = _BatchWrapper([[cam_high_cpu, cam_high_cpu, cam_high_cpu]], model_device)
            
            # Predict value - episode_idx needs to be iterable
            episode_index_tensor = torch.tensor([episode_idx])
            # Monkeypatch torchvision's ToPILImage with a robust converter
            # to avoid channel-order/dtype issues in the loaded ValueFunction.
            from PIL import Image
            import numpy as _np
            import torchvision.transforms as _transforms

            def _to_pil_image_tensor_custom(tensor):
                arr = tensor.detach().cpu().numpy()
                # Handle (C,H,W) -> (H,W,C)
                if arr.ndim == 3 and arr.shape[0] in (1, 3, 4):
                    arr = _np.transpose(arr, (1, 2, 0))
                # Float in [0,1] -> uint8
                if _np.issubdtype(arr.dtype, _np.floating):
                    arr = (_np.clip(arr, 0.0, 1.0) * 255.0).astype(_np.uint8)
                else:
                    arr = arr.astype(_np.uint8)
                return Image.fromarray(arr)

            class _ToPILWrapper:
                def __call__(self, pic):
                    return _to_pil_image_tensor_custom(pic)

            def _ToPILFactory():
                return _ToPILWrapper()

            _transforms.ToPILImage = _ToPILFactory

            value = value_function.predict_value(obs_batch, episode_index_tensor)
            values.append(value.item())
        
        # Store results
        all_results[episode_idx] = {
            'num_steps': len(values),
            'predicted_values': values,
            'true_values': None,  # No true values in LeRobot dataset
        }
        
        # Plot values for this episode
        plot_episode_values(values, episode_idx, args.output_dir)
        
        episode_start_idx += episode_length
    
    # Save summary results to JSON
    summary = {
        'num_episodes': len(all_results),
        'episodes': all_results
    }
    
    summary_path = os.path.join(args.output_dir, "evaluation_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nEvaluation complete!")
    print(f"Results saved to: {args.output_dir}")
    print(f"Summary saved to: {summary_path}")
    print(f"Total episodes evaluated: {len(all_results)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute and plot value function predictions for LeRobot dataset episodes.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the ValueFunction model checkpoint.")
    parser.add_argument("--dataset_repo", type=str, required=True, help="Hugging Face repo ID for the LeRobot dataset.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the output plots.")
    parser.add_argument("--start_episode", type=int, default=None, help="(Optional) start episode index (inclusive).")
    parser.add_argument("--end_episode", type=int, default=None, help="(Optional) end episode index (inclusive).")
    
    args = parser.parse_args()
    main(args)