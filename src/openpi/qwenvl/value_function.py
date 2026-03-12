import sys
# sys.path.append("/project/peilab/wzj/RoboTwin/policy/openpi_test/src/openpi")
# sys.path.append("/project/peilab/wzj/RoboTwin/policy/openpi_test/src/openpi/qwen-vl-utils")
sys.path.append("/project/peilab/wzj/RoboTwin/policy/openpi_test/src/openpi/qwen-vl-utils/src")

import torch
import torch.nn as nn
from transformers import Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info
#from openpi.qwen_vl.qwen_eval import qwen_eval
from transformers import AutoProcessor
from openpi.qwenvl.utils.value_tokenizer import ValueTokenizer
from PIL import Image
from torchvision import transforms
import numpy as np
import lerobot.common.datasets.lerobot_dataset as lerobot_dataset
MIN_VALUE = -1.0
MAX_VALUE = 0.0
BINS = 201
IMAGE_SIZE = (224,224)
def load_model_and_processor(model_name_or_path, attn_implementation=None):
    """Load model and processor from checkpoint."""
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_name_or_path,
        attn_implementation=attn_implementation,
        torch_dtype=torch.bfloat16,
    )

    processor = AutoProcessor.from_pretrained(model_name_or_path)
    model.eval()
    model.config.use_cache = True
    value_tokenizer = ValueTokenizer(
            llm_path=model_name_or_path,  # Use same model path to ensure consistency
            bins=BINS,
            min_value=MIN_VALUE,
            max_value=MAX_VALUE,
    )
    return model, processor, value_tokenizer

class ValueFunction(nn.Module):
    def __init__(self, config):
        #model_path = "/project/peilab/junhao/Value_Function/qwen-vl-finetune/output/gpus_8/checkpoint-3000"
        super().__init__()
        self.model, self.processor, self.value_tokenizer = load_model_and_processor(config.value_function_path)
        self.dataset = lerobot_dataset.LeRobotDataset(config.data.repo_id)
    def get_value(self, obs):
        return self.model.get_value(obs)
    def forward(self, obs_batch):
        pass
    def preprocess_obs(self, obs_batch):
        if obs_batch[0].dtype == torch.float32:
            obs_batch = []
        if isinstance(obs, torch.Tensor):
            obs = obs.to(self.device)
        elif isinstance(obs, np.ndarray):
            obs = torch.from_numpy(obs).to(self.device)
        else:
            raise ValueError(f"Unsupported observation type: {type(obs)}")
        return obs_batch
    @torch.inference_mode()
    def predict_value(self, obs_batch, episode_index):
        device = obs_batch.device
        to_pil = transforms.ToPILImage()
        obs_batch_tmp = []
        for obs_three in obs_batch:
            obs_three = [to_pil(obs.detach().permute(2, 1, 0)) for obs in obs_three]
            obs_batch_tmp.append(obs_three)
        obs_batch = obs_batch_tmp
        instructions = [self.dataset[index.item()]['task'] for index in episode_index]
        messages = process_obs(obs_batch, instructions)
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,  # Add generation prompt to get model to generate
            return_dict=True,
            return_tensors="pt"
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        generation_config = {
            "max_new_tokens": 1,  # We only need one value token
            "do_sample": False,   # Greedy decoding for deterministic prediction
            "pad_token_id": self.processor.tokenizer.pad_token_id,
            "eos_token_id": self.processor.tokenizer.eos_token_id,
        }
        generated_outputs = self.model.generate(
            **inputs,
            **generation_config,
            return_dict_in_generate=True,
            output_logits=True
        )

        interpolated_values = decode_value_token_with_interpolation(self.value_tokenizer, generated_outputs, device)
        return interpolated_values
    
        # generated_token_ids = [sequence[len(inputs['input_ids'][0]):] for sequence in generated_outputs.sequences]
        # generated_token_id = [token_id.item() if len(token_id) > 0 else None for token_id in generated_token_ids]
        # predicted_values = [decode_value_token(self.value_tokenizer, token_id) for token_id in generated_token_id if token_id and token_id in self.value_tokenizer.extra_id_token_ids]
        # return predicted_values

def process_obs(obs_batch, instructions):
    messages = []
    for obs,instruction in zip(obs_batch, instructions):
        message_template = [{
            'role' : 'user',
            'content' : [
                {
                    'type' : 'text',
                    'text' : f""" You are a rigorous, impartial vision evaluator for robot task progress. Given a task instruction and three-views observation images, your job is to estimate the current progress toward accomplishing the task.\n\n# Evaluation Criteria (apply across all three views)\n1) Task Alignment: Evidence directly tied to Task Instruction.\n2) Completeness & Accuracy: Correct pose, contact, placement, orientation, grasp quality, absence of collisions, stability, etc.\n3) View-Specific Evidence & Consistency:\n - Use the **Front** view for global layout, object pose, approach path, end-state geometry, and scene-level constraints.\n - Use the **Left/Right Wrist** views to inspect **fine-grained gripper state** (finger closure, contact location/area, slippage, wedge/misalignment, object deformation, cable/wire/cloth entanglement, unintended contact, occluded collisions).\n - When views disagree, prioritize the view that provides **decisive cues** for the criterion at hand. In particular, wrist views often **override** for grasp/contact validity and safety.\n - If any single view shows a failure that invalidates success (e.g., mis-grasp, collision, unsafe/unstable pose), let that override when judging progress.\n4) Ignore Irrelevant Factors: Lighting, color shifts, background clutter, or UI/watermarks that don't affect task success.\n\nTask Instruction: {instruction}\nRobot Front Image: <image>\nRobot Left Wrist Image: <image>\nRobot Right Wrist Image: <image>\n Progress toward accomplishing the task: """
                },
                {
                    'type' : 'image',
                    'image' : obs[0]
                },
                {
                    'type' : 'image',
                    'image' : obs[1]
                },
                {
                    'type' : 'image',
                    'image' : obs[2]
                }
            ]
        }]
        messages.append(message_template)
    return messages
        
def decode_value_token(value_tokenizer, generated_token_id):
        """Decode a single token ID to continuous value using ValueTokenizer."""
        # Convert token ID to numpy array
        token_id_array = np.array([generated_token_id])
        # Decode using value_tokenizer
        value = value_tokenizer.decode_token_ids_to_values(token_id_array)
        return value[0] if len(value) > 0 else 0.0

def decode_value_token_with_interpolation(value_tokenizer, generated_outputs, device):
    """Decode value tokens with interpolation based on logits for continuous values."""
    # Get the logits for the generated token (last token)
    # logits shape: [batch_size, seq_len, vocab_size]
    logits = generated_outputs.logits[0]  # Get first batch

    if len(logits) == 0:
        return 0.0

    # Get logits for the last generated token
    #last_token_logits = logits[-1]  # Shape: [vocab_size]

    # Extract logits only for value tokens
    value_token_ids = value_tokenizer.extra_id_token_ids

    value_logits = torch.stack([last_token_logits[value_token_ids] for last_token_logits in logits])  # Shape: [n_bins]

    # Apply softmax to get probabilities
    value_probs = torch.softmax(value_logits, dim=1)

    # Interpolate between bin centers using probabilities as weights
    bin_centers = torch.tensor(value_tokenizer.bin_centers, device=device)  # Shape: [n_bins]
    interpolated_value = torch.sum(value_probs * bin_centers, dim=1)

    return interpolated_value