import dataclasses
from pathlib import Path
import shutil
from typing import Literal

import h5py
from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
# from lerobot.common.datasets.push_dataset_to_hub._download_raw import download_raw
import numpy as np
import torch
import tqdm
import tyro
import json
import os
import fnmatch
import pickle
import torch.nn.functional as F
import tqdm
@dataclasses.dataclass(frozen=True)
class DatasetConfig:
    use_videos: bool = True
    tolerance_s: float = 0.0001
    image_writer_processes: int = 10
    image_writer_threads: int = 5
    video_backend: str | None = None


DEFAULT_DATASET_CONFIG = DatasetConfig()
def create_empty_dataset(
    repo_id: str,
    robot_type: str,
    mode: Literal["video", "image"] = "video",
    *,
    has_velocity: bool = False,
    has_effort: bool = False,
    dataset_config: DatasetConfig = DEFAULT_DATASET_CONFIG,
) -> LeRobotDataset:
    motors = [
        "left_waist",
        "left_shoulder",
        "left_elbow",
        "left_forearm_roll",
        "left_wrist_angle",
        "left_wrist_rotate",
        "left_gripper",
        "right_waist",
        "right_shoulder",
        "right_elbow",
        "right_forearm_roll",
        "right_wrist_angle",
        "right_wrist_rotate",
        "right_gripper",
    ]

    cameras = [
        "cam_high",
        "cam_left_wrist",
        "cam_right_wrist",
    ]

    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (len(motors), ),
            "names": [
                motors,
            ],
        },
        "action": {
            "dtype": "float32",
            "shape": (len(motors), ),
            "names": [
                motors,
            ],
        },
        'reward' :{
            'dtype' : 'uint8',
            'shape' : (1, ),
            'names' : ['reward'],
        },
        'result':{
            'dtype' : 'uint8',
            'shape' : (1, ),
            'names' : ['result'],
        }
    }

    # if has_velocity:
    #     features["observation.velocity"] = {
    #         "dtype": "float32",
    #         "shape": (len(motors), ),
    #         "names": [
    #             motors,
    #         ],
    #     }

    # if has_effort:
    #     features["observation.effort"] = {
    #         "dtype": "float32",
    #         "shape": (len(motors), ),
    #         "names": [
    #             motors,
    #         ],
    #     }

    for cam in cameras:
        features[f"observation.images.{cam}"] = {
            "dtype": mode,
            "shape": (3, 224, 224),
            "names": [
                "channels",
                "height",
                "width",
            ],
        }

    if Path(HF_LEROBOT_HOME / repo_id).exists():
        shutil.rmtree(HF_LEROBOT_HOME / repo_id)

    return LeRobotDataset.create(
        repo_id=repo_id,
        fps=20,
        robot_type=robot_type,
        features=features,
        use_videos=dataset_config.use_videos,
        tolerance_s=dataset_config.tolerance_s,
        image_writer_processes=dataset_config.image_writer_processes,
        image_writer_threads=dataset_config.image_writer_threads,
        video_backend=dataset_config.video_backend,
    )

def load_episode_part(file_name):
    try:
        episode_part_file = os.path.join('tmp_data', file_name)
        with open(episode_part_file, 'rb') as f:
            data = pickle.load(f)

            return data['results']

    except FileNotFoundError:
        return None
def main():
    dataset = create_empty_dataset(
        repo_id="beat_block_hammer_rollout",
        robot_type= "aloha",
        mode='image',
        has_effort=True,
        has_velocity=True,
        dataset_config=DEFAULT_DATASET_CONFIG,
    )
    data = load_episode_part('beat_block_hammer_rollout_1.pkl')
    if data is not None:
        for trajectory in tqdm.tqdm(data, desc="Trajectories"):
            for sample in trajectory['trajectory']:
                for image_name in ["cam_high", "cam_left_wrist", "cam_right_wrist"]:
                    sample['observation']['images'][image_name] =F.interpolate(
                            torch.tensor(sample['observation']['images'][image_name]).unsqueeze(0),  # 添加batch维度: (1, 3, 480, 640)
                            size=(224, 224),
                            mode='bilinear',  # 或 'nearest', 'bicubic'
                            align_corners=False
                        ).squeeze(0)  # 移除batch维度: (3, 224, 224)
                tmp_sample = {
                    "observation.state": torch.tensor(sample['observation']['state'], dtype=torch.float32),
                    "observation.images.cam_high": sample['observation']['images']['cam_high'],
                    "observation.images.cam_left_wrist": sample['observation']['images']['cam_left_wrist'],
                    "observation.images.cam_right_wrist": sample['observation']['images']['cam_right_wrist'],
                    "action": torch.tensor(sample['action'], dtype=torch.float32),
                    'reward': torch.tensor([int(sample['reward'])], dtype=torch.uint8),
                    'result': torch.tensor([int(trajectory['result'])], dtype=torch.uint8),
                    'task' : sample['observation']['prompt'],
                }
                dataset.add_frame(tmp_sample)
            dataset.save_episode()

if __name__ == "__main__":
    # import debugpy
    # print("Waiting for debugger to attach")
    # debugpy.listen(("0.0.0.0", 5678))
    # debugpy.wait_for_client()
    # print("Debugger attached")
    main()