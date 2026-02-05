"""
从 lerobot dataset v3.0 转换到 lerobot v2.1 格式的转换脚本
v3.0 使用 video 格式，v2.1 使用 image 格式
需要从视频文件中提取每一帧图像

使用方法:
    python lerobot_v30_to_v21.py --input-repo-id <v3.0_repo_id> --output-repo-id <v2.1_repo_id> --home-lerobot <path>
"""

import os
import json
import shutil
from pathlib import Path
from typing import Optional, Literal, Dict, Any, List
import numpy as np
import pyarrow.parquet as pq
import pyarrow as pa
import tqdm
from collections import defaultdict
import contextlib
import sys

# 抑制 FFmpeg/AV1 警告的环境变量设置
# 在导入 OpenCV 之前设置，确保所有进程都使用这些设置
os.environ['OPENCV_FFMPEG_LOGLEVEL'] = '-8'  # 抑制所有 FFmpeg 日志
os.environ['OPENCV_FFMPEG_READ_ATTEMPTS'] = '1'

# 只使用 v2.1 的 API
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, HF_LEROBOT_HOME

# v3.0 state 中 qpos 的索引位置
QPOS_START_IDX = 52
QPOS_END_IDX = 66
QPOS_DIM = 14


def extract_qpos_from_v30_state(state_v30: np.ndarray) -> np.ndarray:
    """从 v3.0 的 state (96维) 中提取 qpos (14维)"""
    if state_v30.shape[0] < QPOS_END_IDX:
        raise ValueError(f"v3.0 state 维度不足: {state_v30.shape[0]}, 需要至少 {QPOS_END_IDX} 维")
    
    qpos = state_v30[QPOS_START_IDX:QPOS_END_IDX]
    if qpos.dtype != np.float32:
        qpos = qpos.astype(np.float32)
    
    return qpos


@contextlib.contextmanager
def suppress_stderr():
    """上下文管理器：临时抑制 stderr 输出"""
    with open(os.devnull, 'w') as devnull:
        original_stderr = sys.stderr
        try:
            sys.stderr = devnull
            yield
        finally:
            sys.stderr = original_stderr


def load_video_frame(video_path: Path, frame_idx: int, debug: bool = False) -> Optional[np.ndarray]:
    """
    从视频文件中加载指定帧
    
    参数:
        video_path: 视频文件路径
        frame_idx: 帧索引（从0开始，相对于视频文件内的索引）
        debug: 是否输出调试信息
        
    返回:
        图像数组 (H, W, C) RGB 格式，如果失败返回 None
    """
    # 首先尝试使用 imageio（对 AV1 等格式支持更好）
    try:
        import imageio
        reader = imageio.get_reader(str(video_path), 'ffmpeg')
        try:
            frame = reader.get_data(frame_idx)
            reader.close()
            # imageio 返回的是 RGB 格式，直接返回
            return frame
        except (IndexError, RuntimeError) as e:
            reader.close()
            if debug:
                print(f"  调试: imageio 读取失败: {e}，尝试 OpenCV")
        except Exception as e:
            reader.close()
            if debug:
                print(f"  调试: imageio 异常: {e}，尝试 OpenCV")
    except ImportError:
        if debug:
            print(f"  调试: imageio 未安装，使用 OpenCV")
    except Exception as e:
        if debug:
            print(f"  调试: imageio 初始化失败: {e}，尝试 OpenCV")
    
    # 如果 imageio 失败，回退到 OpenCV
    try:
        import cv2
        import warnings
        
        # 抑制 OpenCV 的 AV1 解码警告
        # 设置 OpenCV 日志级别（如果支持）
        try:
            # OpenCV 4.x 支持 setLogLevel
            cv2.setLogLevel(0)  # 0 = SILENT, 1 = ERROR, 2 = WARN, 3 = INFO, 4 = DEBUG
        except AttributeError:
            # 旧版本可能不支持，忽略
            pass
        
        # 使用上下文管理器确保在整个视频操作过程中都抑制警告
        ret = False
        frame = None
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            # 在整个视频操作过程中抑制 stderr（包括 FFmpeg/AV1 警告）
            # 但我们需要捕获一些关键信息用于调试
            cap = None
            try:
                cap = cv2.VideoCapture(str(video_path))
                
                if not cap.isOpened():
                    if debug:
                        print(f"  调试: 无法打开视频文件: {video_path}")
                    return None
                
                # 获取视频属性（用于调试）
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                fps = cap.get(cv2.CAP_PROP_FPS)
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                
                if debug:
                    print(f"  调试: 视频属性 - 总帧数={total_frames}, FPS={fps}, 尺寸={width}x{height}, 请求帧={frame_idx}")
                
                # 检查帧索引是否在有效范围内
                if frame_idx < 0:
                    if debug:
                        print(f"  调试: 帧索引 {frame_idx} 无效（小于0）")
                    cap.release()
                    return None
                
                if total_frames > 0 and frame_idx >= total_frames:
                    if debug:
                        print(f"  调试: 帧索引 {frame_idx} 超出范围（总帧数={total_frames}）")
                    cap.release()
                    return None
                
                # 尝试设置到指定帧并读取
                # 注意：对于某些编码格式（如 AV1），set() 可能成功但 read() 仍然失败
                # 所以我们需要先尝试 set()+read()，如果失败则回退到逐帧读取
                set_pos = cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                
                if set_pos:
                    # set() 成功，尝试读取
                    ret, frame = cap.read()
                    if not ret:
                        if debug:
                            print(f"  调试: set() 成功但 read() 失败，回退到逐帧读取")
                        # read() 失败，回退到逐帧读取
                        set_pos = False
                
                if not set_pos or not ret:
                    if debug and set_pos:
                        print(f"  调试: 回退到逐帧读取")
                    elif debug:
                        print(f"  调试: set() 失败，尝试逐帧读取")
                    # 从开头逐帧读取（较慢但更可靠）
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret = False
                    frame = None
                    for i in range(frame_idx + 1):
                        ret, frame = cap.read()
                        if not ret:
                            if debug:
                                print(f"  调试: 逐帧读取失败，在第 {i} 帧时无法读取")
                            cap.release()
                            return None
                
                cap.release()
                
            except Exception as e:
                if cap is not None:
                    cap.release()
                if debug:
                    print(f"  调试: 异常: {e}")
                return None
        
        if ret and frame is not None:
            # BGR 转 RGB
            try:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                return frame
            except Exception as e:
                if debug:
                    print(f"  调试: BGR转RGB失败: {e}")
                return None
        else:
            if debug:
                print(f"  调试: 读取失败 - ret={ret}, frame is None={frame is None}")
        return None
    except Exception as e:
        if debug:
            print(f"  调试: 外层异常: {e}")
        return None


def find_video_file(input_path: Path, video_key: str, chunk_index: int, file_index: int, video_path_pattern: str) -> Optional[Path]:
    """
    根据视频路径模式查找视频文件
    
    参数:
        input_path: 数据集根路径
        video_key: 视频键（如 "observation.images.cam_high"）
        chunk_index: chunk 索引
        file_index: file 索引
        video_path_pattern: 视频路径模式，如 "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"
        
    返回:
        视频文件路径，如果不存在返回 None
    """
    # 从 video_key 中提取实际的视频目录名
    # 例如 "observation.images.cam_high" -> "cam_high"
    video_dir_name = video_key.replace("observation.images.", "")
    
    try:
        video_path = input_path / video_path_pattern.format(
            video_key=video_dir_name,
            chunk_index=chunk_index,
            file_index=file_index
        )
        
        if video_path.exists():
            return video_path
        
        # 如果找不到，尝试其他可能的路径格式
        # 尝试直接使用 video_key 作为目录名
        alt_video_path = input_path / video_path_pattern.format(
            video_key=video_key,
            chunk_index=chunk_index,
            file_index=file_index
        )
        if alt_video_path.exists():
            return alt_video_path
        
        return None
    except Exception as e:
        return None


def load_v30_info(input_path: Path) -> Dict[str, Any]:
    """加载 v3.0 数据集的 info.json"""
    info_path = input_path / "meta" / "info.json"
    if not info_path.exists():
        raise FileNotFoundError(f"找不到 info.json: {info_path}")
    
    with open(info_path, 'r') as f:
        info = json.load(f)
    
    return info


def extract_features_from_v30_info(info: Dict[str, Any], use_qpos_only: bool = True) -> Dict[str, Any]:
    """
    从 v3.0 的 info.json 中提取特征定义，转换为 v2.1 格式
    v3.0 使用 video，v2.1 使用 image
    """
    features = {}
    v30_features = info.get("features", {})
    
    motors = [
        "left_waist", "left_shoulder", "left_elbow", "left_forearm_roll",
        "left_wrist_angle", "left_wrist_rotate", "left_gripper",
        "right_waist", "right_shoulder", "right_elbow", "right_forearm_roll",
        "right_wrist_angle", "right_wrist_rotate", "right_gripper",
    ]
    
    cameras = ["cam_high", "cam_left_wrist", "cam_right_wrist"]
    
    # 处理 observation.state
    if "observation.state" in v30_features:
        if use_qpos_only:
            features["observation.state"] = {
                "dtype": "float32",
                "shape": (len(motors),),
                "names": [motors],
            }
    
    # 处理 action
    if "action" in v30_features:
        action_info = v30_features["action"]
        features["action"] = {
            "dtype": action_info.get("dtype", "float32"),
            "shape": (len(motors),),
            "names": [motors],
        }
    
    # 注意：task 不是 frame 的特征，而是 save_episode 的参数
    # 所以这里不添加 task 到 features 中
    
    # 处理图像特征 - v3.0 是 video，v2.1 是 image
    # v2.1 的图像格式也是 (height, width, channels)，和 v3.0 一样
    for key, feature_info in v30_features.items():
        if key.startswith("observation.images."):
            shape = feature_info.get("shape", [])
            
            # v2.1 使用 (height, width, channels) 格式，和 v3.0 一样
            if len(shape) == 3:
                new_shape = tuple(shape)  # (480, 640, 3) - 保持原格式
                new_names = ["height", "width", "channel"]
            else:
                new_shape = tuple(shape) if isinstance(shape, list) else shape
                new_names = feature_info.get("names")
            
            features[key] = {
                "dtype": "image",  # v2.1 使用 image
                "shape": new_shape,
                "names": new_names,
            }
    
    return features


def find_all_parquet_files(input_path: Path, data_path_pattern: str) -> List[Path]:
    """查找所有 parquet 文件"""
    parquet_files = []
    data_dir = input_path / "data"
    
    if not data_dir.exists():
        raise FileNotFoundError(f"找不到数据目录: {data_dir}")
    
    for chunk_dir in sorted(data_dir.glob("chunk-*")):
        if chunk_dir.is_dir():
            for parquet_file in sorted(chunk_dir.glob("file-*.parquet")):
                parquet_files.append(parquet_file)
    
    return parquet_files


def parse_chunk_and_file_index(parquet_path: Path) -> tuple[int, int]:
    """
    从 parquet 文件路径中解析 chunk_index 和 file_index
    
    例如: data/chunk-000/file-002.parquet -> (0, 2)
    """
    chunk_dir = parquet_path.parent
    file_name = parquet_path.name
    
    # 从 chunk-000 中提取 000
    chunk_str = chunk_dir.name.replace("chunk-", "")
    chunk_index = int(chunk_str)
    
    # 从 file-002.parquet 中提取 002
    file_str = file_name.replace("file-", "").replace(".parquet", "")
    file_index = int(file_str)
    
    return chunk_index, file_index


def read_parquet_file(parquet_path: Path) -> pa.Table:
    """读取 parquet 文件"""
    return pq.read_table(parquet_path)


def extract_episode_index(row_dict: Dict[str, Any]) -> Optional[int]:
    """从行数据中提取 episode_index"""
    if 'episode_index' in row_dict:
        ep_val = row_dict['episode_index']
        
        # 处理各种可能的类型
        if ep_val is None:
            return None
        
        # 如果是 Python 基本类型（int），直接返回
        if isinstance(ep_val, int):
            return int(ep_val)
        
        # 处理 numpy 类型
        if isinstance(ep_val, np.ndarray):
            if ep_val.size > 0:
                return int(ep_val.item() if ep_val.size == 1 else ep_val[0])
        
        # 处理列表/元组
        if isinstance(ep_val, (list, tuple)):
            if len(ep_val) > 0:
                return int(ep_val[0])
        
        # 处理 numpy integer 类型
        if isinstance(ep_val, np.integer):
            return int(ep_val)
        
        # 处理 pyarrow Scalar
        if isinstance(ep_val, pa.Scalar):
            if ep_val.is_valid:
                return int(ep_val.as_py())
        
        # 处理其他可能有 as_py 方法的对象
        if hasattr(ep_val, 'as_py'):
            try:
                return int(ep_val.as_py())
            except:
                pass
        
        # 尝试直接转换
        try:
            return int(ep_val)
        except (ValueError, TypeError):
            pass
    
    return None


def extract_frame_index(row_dict: Dict[str, Any]) -> Optional[int]:
    """从行数据中提取 frame_index"""
    if 'frame_index' in row_dict:
        frame_val = row_dict['frame_index']
        if isinstance(frame_val, (np.ndarray, list, pa.Array)):
            if len(frame_val) > 0:
                frame_idx = int(frame_val[0] if hasattr(frame_val, '__getitem__') else frame_val)
                return frame_idx
        elif isinstance(frame_val, (int, np.integer)):
            return int(frame_val)
        elif hasattr(frame_val, 'as_py'):
            return int(frame_val.as_py())
    
    return None


def convert_table_to_dicts(table: pa.Table) -> List[Dict[str, Any]]:
    """将 pyarrow Table 转换为字典列表"""
    num_rows = len(table)
    rows = []
    
    for i in range(num_rows):
        row_dict = {}
        for column_name in table.column_names:
            column = table[column_name]
            value = column[i]
            
            # 处理 pyarrow 类型
            # 优先检查是否是 Array 类型（包括 ListArray），这些需要转换为 numpy array
            if isinstance(value, pa.Array):
                # pyarrow Array（包括 ListArray），转换为 numpy
                # 对于 ListArray，to_numpy() 可能返回 Python list，需要手动转换
                if isinstance(value.type, pa.ListType):
                    # ListArray：转换为 Python list，然后转换为 numpy array
                    value_list = value.to_pylist()
                    if len(value_list) > 0 and isinstance(value_list[0], list):
                        # 嵌套列表，取第一个元素
                        value = np.array(value_list[0], dtype=np.float32)
                    else:
                        value = np.array(value_list, dtype=np.float32)
                else:
                    # 普通 Array，直接转换为 numpy
                    value = value.to_numpy()
                    # 确保是 numpy array（to_numpy() 可能返回 Python list）
                    if not isinstance(value, np.ndarray):
                        value = np.array(value)
                
                # 只有索引字段才提取标量值
                if column_name in ['episode_index', 'frame_index', 'index', 'task_index', 'timestamp', 'time_stamp']:
                    if isinstance(value, np.ndarray) and value.size == 1:
                        value = value.item()
            # 然后检查是否是 Scalar 类型（包括 Int64Scalar 等）
            elif hasattr(value, 'as_py') and hasattr(value, 'is_valid'):
                # 所有 pyarrow Scalar 类型都有 as_py 和 is_valid 方法
                try:
                    if not value.is_valid:
                        value = None
                    else:
                        value = value.as_py()
                except:
                    # 如果 as_py 失败，尝试其他方法
                    try:
                        value = int(value) if hasattr(value, '__int__') else value
                    except:
                        value = value
            elif isinstance(value, pa.Scalar):
                # 标准的 pa.Scalar（备用检查）
                if value.is_valid:
                    value = value.as_py()
                else:
                    value = None
            elif isinstance(value, (list, tuple)):
                # 列表/元组转换为 numpy array
                value = np.array(value)
                # 只有索引字段才提取标量值
                if column_name in ['episode_index', 'frame_index', 'index', 'task_index']:
                    if value.size == 1:
                        value = value.item()
            elif hasattr(value, 'as_py'):
                # 其他有 as_py 方法的对象
                value = value.as_py()
            elif isinstance(value, (int, float, str, bool)) or value is None:
                # 已经是 Python 基本类型，保持不变
                pass
            else:
                # 尝试转换为 Python 类型
                try:
                    if hasattr(value, '__int__'):
                        value = int(value)
                    elif hasattr(value, '__float__'):
                        value = float(value)
                except:
                    pass
            
            row_dict[column_name] = value
        
        rows.append(row_dict)
    
    return rows


def convert_v30_to_v21(
    input_repo_id: str,
    output_repo_id: str,
    home_lerobot: Optional[str] = None,
    robot_type: str = "franka",
    fps: Optional[int] = None,
    use_videos: bool = False,  # v2.1 使用 image，所以这里应该是 False
    image_writer_processes: int = 10,
    image_writer_threads: int = 5,
    video_backend: Optional[str] = None,
    use_qpos_only: bool = True,
):
    """
    将 lerobot v3.0 数据集转换为 v2.1 格式
    v3.0 使用 video，v2.1 使用 image
    需要从视频文件中提取每一帧
    """
    # 确保环境变量已设置（用于多进程环境）
    os.environ['OPENCV_FFMPEG_LOGLEVEL'] = '-8'
    os.environ['OPENCV_FFMPEG_READ_ATTEMPTS'] = '1'
    
    # 设置路径
    if home_lerobot is None:
        home_lerobot = str(HF_LEROBOT_HOME)
    else:
        home_lerobot = str(Path(home_lerobot))
    
    input_path = Path(home_lerobot) / input_repo_id
    output_path = HF_LEROBOT_HOME / output_repo_id
    
    if not input_path.exists():
        raise FileNotFoundError(f"找不到输入数据集: {input_path}")
    
    # 加载 v3.0 数据集信息
    print(f"加载 v3.0 数据集信息: {input_repo_id}")
    info_v30 = load_v30_info(input_path)
    
    if fps is None:
        fps = info_v30.get("fps", 20)
    
    video_path_pattern = info_v30.get("video_path", "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4")
    
    print(f"数据集 FPS: {fps}")
    print(f"总 episode 数: {info_v30.get('total_episodes', 'unknown')}")
    print(f"总帧数: {info_v30.get('total_frames', 'unknown')}")
    print(f"视频路径模式: {video_path_pattern}")
    
    # 提取特征定义
    print("提取特征定义...")
    if use_qpos_only:
        print("只提取 qpos (14维) 作为 state")
    features = extract_features_from_v30_info(info_v30, use_qpos_only=use_qpos_only)
    
    print(f"找到 {len(features)} 个特征:")
    for key in features.keys():
        print(f"  - {key}")
    
    # 创建 v2.1 数据集（使用 image 模式）
    print(f"创建 v2.1 数据集: {output_repo_id} (使用 image 格式)")
    if output_path.exists():
        print(f"删除已存在的输出目录: {output_path}")
        shutil.rmtree(output_path)
    
    dataset_v21 = LeRobotDataset.create(
        repo_id=output_repo_id,
        fps=fps,
        robot_type=robot_type,
        features=features,
        use_videos=use_videos,  # False，使用 image
        tolerance_s=0.0001,
        image_writer_processes=image_writer_processes,
        image_writer_threads=image_writer_threads,
        video_backend=video_backend,
    )
    
    # 查找所有 parquet 文件
    data_path_pattern = info_v30.get("data_path", "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet")
    print(f"查找 parquet 文件...")
    parquet_files = find_all_parquet_files(input_path, data_path_pattern)
    print(f"找到 {len(parquet_files)} 个 parquet 文件")
    
    # 测试读取第一个 parquet 文件的结构（用于调试）
    if len(parquet_files) > 0:
        print(f"\n测试读取第一个 parquet 文件的结构: {parquet_files[0].name}")
        test_table = read_parquet_file(parquet_files[0])
        print(f"  Parquet 文件列名: {test_table.column_names[:10]}...")  # 显示前10个列名
        if 'episode_index' in test_table.column_names:
            test_col = test_table['episode_index']
            print(f"  episode_index 列类型: {test_col.type}")
            if len(test_col) > 0:
                test_val = test_col[0]
                print(f"  第一个值: {test_val}, 类型: {type(test_val)}")
                if isinstance(test_val, pa.Scalar):
                    print(f"    Scalar 值: {test_val.as_py()}")
                elif isinstance(test_val, pa.Array):
                    print(f"    Array 值: {test_val.to_numpy()}")
        
        # 测试视频文件路径
        print(f"\n测试视频文件路径:")
        test_chunk, test_file = parse_chunk_and_file_index(parquet_files[0])
        for cam in ["cam_high", "cam_left_wrist", "cam_right_wrist"]:
            test_video_key = f"observation.images.{cam}"
            test_video_file = find_video_file(input_path, test_video_key, test_chunk, test_file, video_path_pattern)
            if test_video_file:
                print(f"  {cam}: {test_video_file} (存在: {test_video_file.exists()})")
            else:
                # 尝试打印可能的路径
                video_dir_name = cam
                possible_path = input_path / video_path_pattern.format(
                    video_key=video_dir_name,
                    chunk_index=test_chunk,
                    file_index=test_file
                )
                print(f"  {cam}: 未找到 (尝试路径: {possible_path})")
        print()
    
    # 按 episode 组织数据
    episode_data = defaultdict(list)
    
    # 处理所有 parquet 文件
    print("读取 parquet 文件并从视频中提取图像...")
    total_rows = 0
    skipped_rows = 0
    video_load_errors = 0
    
    for parquet_file in tqdm.tqdm(parquet_files, desc="处理 parquet 文件"):
        # 解析 chunk_index 和 file_index
        chunk_index, file_index = parse_chunk_and_file_index(parquet_file)
        
        table = read_parquet_file(parquet_file)
        rows = convert_table_to_dicts(table)
        total_rows += len(rows)
        
        # 测试第一行的转换结果（用于调试）- 只对第一个文件的第一行进行测试
        if parquet_file == parquet_files[0] and len(rows) > 0:
            test_row = rows[0]
            print(f"\n测试第一行转换结果 (文件: {parquet_file.name}):")
            if 'episode_index' in test_row:
                ep_val = test_row['episode_index']
                print(f"  episode_index 值: {ep_val}, 类型: {type(ep_val)}")
                test_ep = extract_episode_index(test_row)
                print(f"  extract_episode_index 结果: {test_ep}")
                if test_ep is None:
                    print(f"  ⚠️ 警告: extract_episode_index 返回 None!")
            else:
                print(f"  ⚠️ 警告: 行中没有 episode_index 键")
                print(f"  行中的键: {list(test_row.keys())}")
            if 'observation.state' in test_row:
                state_val = test_row['observation.state']
                print(f"  observation.state 类型: {type(state_val)}")
                if isinstance(state_val, np.ndarray):
                    print(f"    shape: {state_val.shape}, dtype: {state_val.dtype}")
                elif isinstance(state_val, list):
                    print(f"    list 长度: {len(state_val) if state_val else 0}")
                    print(f"    前5个元素: {state_val[:5] if len(state_val) > 5 else state_val}")
                else:
                    print(f"    ⚠️ 不是 np.ndarray，实际类型: {type(state_val)}")
        
        for row_idx, row in enumerate(rows):
            # 获取 episode_index
            episode_idx = extract_episode_index(row)
            
            if episode_idx is None:
                skipped_rows += 1
                if skipped_rows <= 10:
                    # 打印调试信息
                    print(f"\n警告: 无法确定 episode_index，跳过该行 (文件: {parquet_file.name}, 行: {row_idx})")
                    if 'episode_index' in row:
                        ep_val = row['episode_index']
                        print(f"  episode_index 值: {ep_val}, 类型: {type(ep_val)}")
                        # 尝试直接转换
                        try:
                            test_ep = int(ep_val) if ep_val is not None else None
                            print(f"  尝试直接转换: {test_ep}")
                        except Exception as e:
                            print(f"  直接转换失败: {e}")
                    else:
                        print(f"  行中的键: {list(row.keys())[:10]}...")  # 只显示前10个键
                continue
            
            # 获取 frame_index
            frame_idx = extract_frame_index(row)
            
            # 构建 frame 数据
            frame = {}
            
            # 处理 observation.state - 只提取 qpos
            if 'observation.state' in row and use_qpos_only:
                state_v30 = row['observation.state']
                # 如果是 list，转换为 numpy array
                if isinstance(state_v30, list):
                    state_v30 = np.array(state_v30, dtype=np.float32)
                elif not isinstance(state_v30, np.ndarray):
                    # 尝试转换为 numpy array
                    try:
                        state_v30 = np.array(state_v30, dtype=np.float32)
                    except:
                        skipped_rows += 1
                        if skipped_rows <= 5:
                            print(f"警告: 无法转换 observation.state，类型: {type(state_v30)}")
                        continue
                
                if isinstance(state_v30, np.ndarray):
                    qpos = extract_qpos_from_v30_state(state_v30)
                    frame['observation.state'] = qpos
                else:
                    skipped_rows += 1
                    continue
            
            # 处理 action
            if 'action' in row:
                action = row['action']
                if isinstance(action, (np.ndarray, list)):
                    if isinstance(action, list):
                        action = np.array(action)
                    if action.dtype == np.float64:
                        action = action.astype(np.float32)
                    frame['action'] = action
            
            # 注意：task 不是 frame 的特征，而是 save_episode 的参数
            # 我们会在保存 episode 时传递 task 参数
            # 这里先提取 task 信息，稍后在保存 episode 时使用
            # 暂时不添加到 frame 中
            
            # 处理图像字段 - 从视频文件中加载
            # 需要确保所有定义的图像特征都被添加到 frame 中
            for key in features.keys():
                if key.startswith('observation.images.'):
                    # 查找对应的视频文件
                    video_file = find_video_file(input_path, key, chunk_index, file_index, video_path_pattern)
                    
                    if video_file and video_file.exists():
                        # 从视频中加载对应帧
                        # 计算在视频文件中的帧索引
                        # 注意：frame_idx 是全局帧索引，而每个视频文件只包含部分帧
                        # 在 v3.0 格式中，每个 parquet 文件对应一个视频文件，
                        # parquet 文件中的每一行对应视频文件中的一帧
                        # 所以应该使用 row_idx（在 parquet 文件中的行索引）作为视频文件内的帧索引
                        video_frame_idx = row_idx  # 使用 parquet 文件内的行索引，而不是全局 frame_idx
                        
                        # 在前几次失败时输出详细调试信息
                        debug_mode = (video_load_errors < 3)
                        image = load_video_frame(video_file, video_frame_idx, debug=debug_mode)
                        if image is not None:
                            # v2.1 使用 (H, W, C) 格式，和 v3.0 一样，不需要转换
                            # image 已经是 (H, W, C) 格式
                            frame[key] = image
                        else:
                            video_load_errors += 1
                            if video_load_errors <= 10:
                                print(f"警告: 无法从视频加载帧: {key}, chunk={chunk_index}, file={file_index}, frame_idx={video_frame_idx}, 视频路径={video_file}")
                                if debug_mode:
                                    print(f"  视频文件存在: {video_file.exists()}, 大小: {video_file.stat().st_size if video_file.exists() else 0} 字节")
                    else:
                        video_load_errors += 1
                        if video_load_errors <= 10:
                            print(f"警告: 找不到视频文件: {key}, chunk={chunk_index}, file={file_index}")
                            if video_file:
                                print(f"  尝试的路径: {video_file}")
                            else:
                                # 尝试打印可能的路径
                                video_dir_name = key.replace("observation.images.", "")
                                possible_path = input_path / video_path_pattern.format(
                                    video_key=video_dir_name,
                                    chunk_index=chunk_index,
                                    file_index=file_index
                                )
                                print(f"  期望的路径: {possible_path}")
            
            # 提取 task 信息（用于 save_episode）
            # 从第一帧提取 task，或者使用默认值
            task_value = "default_task"
            if 'task_index' in row:
                task_idx = row.get('task_index', 0)
                if isinstance(task_idx, (list, np.ndarray)):
                    task_idx = int(task_idx[0] if len(task_idx) > 0 else 0)
                elif isinstance(task_idx, (int, np.integer)):
                    task_idx = int(task_idx)
                else:
                    try:
                        task_idx = int(task_idx)
                    except:
                        task_idx = 0
                task_value = f"task_{task_idx}"
            
            # 存储 frame、frame_index 和 task（用于排序和保存）
            episode_data[episode_idx].append((frame_idx, frame, task_value))
    
    print(f"总共读取 {total_rows} 行数据，跳过 {skipped_rows} 行")
    print(f"视频加载错误: {video_load_errors} 次")
    print(f"找到 {len(episode_data)} 个不同的 episode")
    
    # 按 episode 顺序保存数据
    print("保存 episode 数据...")
    sorted_episodes = sorted(episode_data.keys())
    
    for episode_idx in tqdm.tqdm(sorted_episodes, desc="保存 episode"):
        frames_with_indices = episode_data[episode_idx]
        
        # 按 frame_index 排序
        if any(idx is not None for idx, _, _ in frames_with_indices):
            frames_with_indices.sort(key=lambda x: x[0] if x[0] is not None else 0)
        
        # 获取 task（从第一帧获取，所有帧应该使用相同的 task）
        task_value = frames_with_indices[0][2] if len(frames_with_indices) > 0 else "default_task"
        
        # 添加所有 frame
        for frame_idx, frame, _ in frames_with_indices:
            # 检查 frame 是否包含所有必需的特征（排除自动添加的索引字段）
            # v2.1 会自动添加 timestamp, frame_index, episode_index, index, task_index
            auto_features = {'timestamp', 'frame_index', 'episode_index', 'index', 'task_index'}
            required_features = set(features.keys()) - auto_features
            missing_features = required_features - set(frame.keys())
            
            if missing_features:
                print(f"\n⚠️ 警告: Frame 缺少特征 (episode {episode_idx}, frame {frame_idx})")
                print(f"缺少的特征: {missing_features}")
                print(f"Frame 中的键: {list(frame.keys())}")
                # 跳过这个 frame，或者尝试添加占位符
                # 对于图像，如果加载失败，我们无法继续
                if any(key.startswith('observation.images.') for key in missing_features):
                    print(f"  缺少图像特征，跳过该 frame")
                    continue
            
            try:
                # 使用 task 参数（如果需要）
                # task 是可选的，如果 features 中没有定义 task，就不需要传递
                dataset_v21.add_frame(frame)
            except ValueError as e:
                # 捕获特征不匹配错误，打印详细信息
                print(f"\n❌ 错误: 添加 frame 时特征不匹配 (episode {episode_idx}, frame {frame_idx})")
                print(f"错误信息: {e}")
                print(f"Frame 中的键: {list(frame.keys())}")
                for key, value in frame.items():
                    if isinstance(value, np.ndarray):
                        print(f"  {key}: shape={value.shape}, dtype={value.dtype}")
                    else:
                        print(f"  {key}: type={type(value)}, value={str(value)[:100]}")
                print(f"期望的特征: {list(dataset_v21.features.keys())}")
                raise
        
        # 保存 episode，传递 task 参数
        dataset_v21.save_episode(task=task_value)
    
    # 合并数据集（v2.1 可能不需要 consolidate，如果需要可以取消注释）
    # print("合并数据集...")
    # dataset_v21.consolidate()
    
    print(f"\n转换完成!")
    print(f"输入数据集: {input_repo_id}")
    print(f"输出数据集: {output_repo_id}")
    print(f"总共处理了 {len(sorted_episodes)} 个 episode")
    print(f"输出路径: {output_path}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="将 lerobot v3.0 (video) 转换为 v2.1 (image) 格式")
    parser.add_argument("--input-repo-id", type=str, required=True, help="v3.0 数据集的 repo_id")
    parser.add_argument("--output-repo-id", type=str, required=True, help="输出的 v2.1 数据集的 repo_id")
    parser.add_argument("--home-lerobot", type=str, default=None, help="lerobot 数据集的本地路径")
    parser.add_argument("--robot-type", type=str, default="franka", help="机器人类型")
    parser.add_argument("--fps", type=int, default=None, help="帧率")
    parser.add_argument("--image-writer-processes", type=int, default=10, help="图像写入进程数")
    parser.add_argument("--image-writer-threads", type=int, default=5, help="图像写入线程数")
    parser.add_argument("--video-backend", type=str, default=None, help="视频后端")
    parser.add_argument("--use-full-state", action="store_true", help="使用完整 state（96维），而不是只提取 qpos")
    
    args = parser.parse_args()
    
    convert_v30_to_v21(
        input_repo_id=args.input_repo_id,
        output_repo_id=args.output_repo_id,
        home_lerobot=args.home_lerobot,
        robot_type=args.robot_type,
        fps=args.fps,
        use_videos=False,  # v2.1 使用 image
        image_writer_processes=args.image_writer_processes,
        image_writer_threads=args.image_writer_threads,
        video_backend=args.video_backend,
        use_qpos_only=not args.use_full_state,
    )