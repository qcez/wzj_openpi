FEATURESV30 = {
    "codebase_version": "v3.0",
    "robot_type": "franka",
    "total_episodes": 250,
    "total_frames": 430497,
    "total_tasks": 1,
    "chunks_size": 1000,
    "data_files_size_in_mb": 100,
    "video_files_size_in_mb": 500,
    "fps": 20,
    "splits": {
        "train": "0:250"
    },
    "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
    "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
    "features": {
        "observation.images.cam_high": {
            "dtype": "video",
            "shape": [
                480,
                640,
                3
            ],
            "names": [
                "height",
                "width",
                "rgb"
            ],
            "info": {
                "video.height": 480,
                "video.width": 640,
                "video.codec": "h264",
                "video.pix_fmt": "yuv420p",
                "video.is_depth_map": False,
                "video.fps": 20,
                "video.channels": 3,
                "has_audio": False
            }
        },
        "observation.images.cam_left_wrist": {
            "dtype": "video",
            "shape": [
                480,
                640,
                3
            ],
            "names": [
                "height",
                "width",
                "rgb"
            ],
            "info": {
                "video.height": 480,
                "video.width": 640,
                "video.codec": "h264",
                "video.pix_fmt": "yuv420p",
                "video.is_depth_map": False,
                "video.fps": 20,
                "video.channels": 3,
                "has_audio": False
            }
        },
        "observation.images.cam_right_wrist": {
            "dtype": "video",
            "shape": [
                480,
                640,
                3
            ],
            "names": [
                "height",
                "width",
                "rgb"
            ],
            "info": {
                "video.height": 480,
                "video.width": 640,
                "video.codec": "h264",
                "video.pix_fmt": "yuv420p",
                "video.is_depth_map": False,
                "video.fps": 20,
                "video.channels": 3,
                "has_audio": False
            }
        },
        "observation.state": {
            "dtype": "float32",
            "shape": [
                96
            ],
            "names": [
                "eef_euler_0",
                "eef_euler_1",
                "eef_euler_2",
                "eef_euler_3",
                "eef_euler_4",
                "eef_euler_5",
                "eef_euler_6",
                "eef_euler_7",
                "eef_euler_8",
                "eef_euler_9",
                "eef_euler_10",
                "eef_euler_11",
                "eef_euler_12",
                "eef_euler_13",
                "eef_quat_0",
                "eef_quat_1",
                "eef_quat_2",
                "eef_quat_3",
                "eef_quat_4",
                "eef_quat_5",
                "eef_quat_6",
                "eef_quat_7",
                "eef_quat_8",
                "eef_quat_9",
                "eef_quat_10",
                "eef_quat_11",
                "eef_quat_12",
                "eef_quat_13",
                "eef_quat_14",
                "eef_quat_15",
                "eef6d_0",
                "eef6d_1",
                "eef6d_2",
                "eef6d_3",
                "eef6d_4",
                "eef6d_5",
                "eef6d_6",
                "eef6d_7",
                "eef6d_8",
                "eef6d_9",
                "eef6d_10",
                "eef6d_11",
                "eef6d_12",
                "eef6d_13",
                "eef6d_14",
                "eef6d_15",
                "eef6d_16",
                "eef6d_17",
                "eef6d_18",
                "eef6d_19",
                "eef_left_time",
                "eef_right_time",
                "qpos_0",
                "qpos_1",
                "qpos_2",
                "qpos_3",
                "qpos_4",
                "qpos_5",
                "qpos_6",
                "qpos_7",
                "qpos_8",
                "qpos_9",
                "qpos_10",
                "qpos_11",
                "qpos_12",
                "qpos_13",
                "qvel_0",
                "qvel_1",
                "qvel_2",
                "qvel_3",
                "qvel_4",
                "qvel_5",
                "qvel_6",
                "qvel_7",
                "qvel_8",
                "qvel_9",
                "qvel_10",
                "qvel_11",
                "qvel_12",
                "qvel_13",
                "effort_0",
                "effort_1",
                "effort_2",
                "effort_3",
                "effort_4",
                "effort_5",
                "effort_6",
                "effort_7",
                "effort_8",
                "effort_9",
                "effort_10",
                "effort_11",
                "effort_12",
                "effort_13",
                "qpos_left_time",
                "qpos_right_time"
            ]
        },
        "action": {
            "dtype": "float32",
            "shape": [
                14
            ],
            "names": {
                "motors": [
                    "joint_action_0",
                    "joint_action_1",
                    "joint_action_2",
                    "joint_action_3",
                    "joint_action_4",
                    "joint_action_5",
                    "joint_action_6",
                    "joint_action_7",
                    "joint_action_8",
                    "joint_action_9",
                    "joint_action_10",
                    "joint_action_11",
                    "joint_action_12",
                    "joint_action_13"
                ]
            }
        },
        "timestamp": {
            "dtype": "float32",
            "shape": [
                1
            ],
            "names": None
        },
        "time_stamp": {
            "dtype": "float32",
            "shape": [
                1
            ],
            "names": {
                "values": [
                    "global_timestamp"
                ]
            }
        },
        "frame_index": {
            "dtype": "int64",
            "shape": [
                1
            ],
            "names": None
        },
        "episode_index": {
            "dtype": "int64",
            "shape": [
                1
            ],
            "names": None
        },
        "index": {
            "dtype": "int64",
            "shape": [
                1
            ],
            "names": None
        },
        "task_index": {
            "dtype": "int64",
            "shape": [
                1
            ],
            "names": None
        }
    }
}


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

FEATURESV21 = {
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
        }

for cam in cameras:
    FEATURESV21[f"observation.images.{cam}"] = {
        "dtype": "image",
        "shape": (3, 480, 640),
        "names": [
            "channels",
            "height",
            "width",
        ],
    }