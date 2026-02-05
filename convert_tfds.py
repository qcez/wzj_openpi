import tensorflow_datasets as tfds
from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME, LeRobotDataset
import numpy as np
#import tensorflow as tf
def get_dataset(in_dir, out_dir):
    import os
    import shutil
    if os.path.exists(os.path.join(HF_LEROBOT_HOME, "coffeev2.1")):
        shutil.rmtree(os.path.join(HF_LEROBOT_HOME, "coffeev2.1"))
    lerobot_dataset = LeRobotDataset.create(
        repo_id="coffeev2.1",
        robot_type="franka",  # 你的机器人类型
        fps=20,              # 帧率
        features={
            "observation.images.wrist_image": {
                "dtype": "video",
                "shape": (256,256,3),
                "names": [
                    "height",
                    "width",
                    "rgb"
                ],
                "info": {
                    "video.height": 256,
                    "video.width": 256,
                    "video.codec": "av1",
                    "video.pix_fmt": "yuv420p",
                    "video.is_depth_map": False,
                    "video.fps": 20,
                    "video.channels": 3,
                    "has_audio": False
                }
            },
            "observation.images.image": {
                "dtype": "video",
                "shape": (256,256,3),
                "names": [
                    "height",
                    "width",
                    "rgb"
                ],
                "info": {
                    "video.height": 256,
                    "video.width": 256,
                    "video.codec": "av1",
                    "video.pix_fmt": "yuv420p",
                    "video.is_depth_map": False,
                    "video.fps": 20,
                    "video.channels": 3,
                    "has_audio": False
                }
            },
            "observation.state": {
                "dtype": "float32",
                "shape": (8,),
                "names": {
                    "motors": [
                        "x",
                        "y",
                        "z",
                        "axis_angle1",
                        "axis_angle2",
                        "axis_angle3",
                        "gripper",
                        "gripper"
                    ]
                }
            },
            "action": {
                "dtype": "float32",
                "shape": (7,),
                "names": {
                    "motors": [
                        "x",
                        "y",
                        "z",
                        "axis_angle1",
                        "axis_angle2",
                        "axis_angle3",
                        "gripper"
                    ]
                }
            }
        }
    )


    dataset = tfds.load(
        split="train",
        name="coffee:1.0.0",
        download=False,
        data_dir = in_dir)
    episodes = dataset.take(300)
    for episode_idx, episode in enumerate(episodes):
        #print(episode["episode_metadata"])
        steps = episode["steps"]
        for step in steps:
            wrist_images = np.asarray(step["observation"]["wrist_image"])
            front_images = np.asarray(step["observation"]["image"])
            states       = np.asarray(step["observation"]["state"])
            states = np.concatenate([states, states[6:7]])      # (8,)
            actions    = np.asarray(step["action"])
            #actions = np.concatenate([actions, actions[6:7]])  # (7,)
            lerobot_dataset.add_frame(
                {
                    "observation.images.wrist_image": wrist_images,
                    "observation.images.image": front_images,
                    "observation.state": states,
                    "action": actions,
                }
            )
    lerobot_dataset.save_episode()
    print("完成第", episode_idx, "个episode")
if __name__ == "__main__":
    get_dataset("/project/peilab/yanzhengyang/RoboTwin/policy/yzy_openpi/processed_data", "/project/peilab/yanzhengyang/RoboTwin/policy/yzy_openpi/processed_data/coffeev2.1")