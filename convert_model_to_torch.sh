#下载原模型
uv run python -c "
from openpi.shared import download

checkpoint_dir = download.maybe_download('gs://openpi-assets/checkpoints/pi0_base')
print('Downloaded to:', checkpoint_dir)"
#转换
uv run examples/convert_jax_model_to_pytorch.py \
    --checkpoint_dir /home/rouyangaa/.cache/openpi/openpi-assets/checkpoints/pi0_base \
    --config_name pi0_base \
    --output_path ~/openpi_checkpoints/pi0_base_pytorch
#验证
uv run python -c "
from openpi.policies import policy_config
from openpi.training import config as _config

config = _config.get_config('pi0_base')
policy = policy_config.create_trained_policy(
    config,
    '/project/peilab/wzj/RoboTwin/policy/openpi_test/base_model_pytorch/pi0_base_pytorch'   # ← 改成你的输出路径
)

print('Successfully loaded PyTorch pi0_base!')
"

uv run python -c "
import dataclasses
from openpi.policies import policy_config
from openpi.training import config as _config
from openpi.training.config import AssetsConfig

# 先加载原始 pi0_base config
config = _config.get_config('pi0_base')

# 创建新的 AssetsConfig，使用官方的 'trossen' (ALOHA/Trossen norm stats，最匹配 pi0_base)
new_assets = AssetsConfig(
    assets_dir='gs://openpi-assets/checkpoints/pi0_base/assets',
    asset_id='trossen'
)

# 创建新的 data config（假设是 LeRobotAlohaDataConfig 或类似，根据你的 config 类型）
# 如果不确定类型，先打印 config.data 看是什么类
new_data = dataclasses.replace(
    config.data,
    assets=new_assets
)

# 再创建新的完整 config
new_config = dataclasses.replace(config, data=new_data)

# 现在加载 policy
policy = policy_config.create_trained_policy(
    new_config,
    '/project/peilab/wzj/RoboTwin/policy/openpi_test/base_model_pytorch/pi0_base_pytorch'
)
print('PyTorch pi0_base 加载成功！使用了 trossen norm stats')
"