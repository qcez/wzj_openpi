import torch
from transformers import Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info
#from openpi.qwen_vl.qwen_eval import qwen_eval
from transformers import AutoProcessor
from openpi.qwenvl.utils.value_tokenizer import ValueTokenizer
from PIL import Image
import numpy as np
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
        device_map="auto",
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

class ValueFunction:
    def __init__(self, model_path):
        #model_path = "/project/peilab/junhao/Value_Function/qwen-vl-finetune/output/gpus_8/checkpoint-3000"
        self.model, self.processor, self.value_tokenizer = load_model_and_processor(model_path)

    def decode_value_token(value_tokenizer, generated_token_id):
        """Decode a single token ID to continuous value using ValueTokenizer."""
        # Convert token ID to numpy array
        token_id_array = np.array([generated_token_id])
        # Decode using value_tokenizer
        value = value_tokenizer.decode_token_ids_to_values(token_id_array)
        return value[0] if len(value) > 0 else 0.0
    def get_value(self, obs):
        return self.model.get_value(obs)
    
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
    def predict_value(self, obs_batch):
        pass