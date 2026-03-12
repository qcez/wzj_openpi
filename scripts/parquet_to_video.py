#!/usr/bin/env python3
"""把 LeRobot/rollout parquet 文件（单个 episode）转换为视频的实用脚本。

用法示例:
  python scripts/parquet_to_video.py \
    --parquet /path/to/episode_000000.parquet \
    --out episode.mp4 --camera observation.images.cam_high --fps 10

脚本能处理以下常见情况：
 - 列保存的是图片 bytes（jpeg/png）
 - 列保存为 Python list / numpy array（C,H,W 或 H,W,C）
 - 列保存为 base64 字符串（会尝试解码）
 支持把三路相机横向合成（--compose 三个列名，按顺序）。
"""
import argparse
import os
from io import BytesIO
import base64
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import datasets
from lerobot.common.datasets.utils import hf_transform_to_torch

try:
    import imageio.v3 as iio
    _USE_IIO = True
except Exception:
    import imageio
    _USE_IIO = False
try:
    from torchvision.transforms import ToPILImage
    _HAS_TORCHVISION = True
except Exception:
    _HAS_TORCHVISION = False


def _to_pil_from_obj(obj):
    """将单元格对象转换为 PIL.Image（RGB）。支持 bytes、base64 字符串、list/ndarray。
    返回 PIL.Image 实例。
    """
    # bytes: png/jpg binary
    if isinstance(obj, (bytes, bytearray)):
        return Image.open(BytesIO(obj)).convert('RGB')

    # base64 字符串
    if isinstance(obj, str):
        # 可能是 base64 编码的图片
        try:
            b = base64.b64decode(obj)
            return Image.open(BytesIO(b)).convert('RGB')
        except Exception:
            # 也可能是 repr(list) 等，尝试 eval 失败则抛出
            try:
                parsed = eval(obj)
            except Exception:
                raise ValueError('无法解析字符串图像单元：非 base64，也非可解析的列表')
            obj = parsed

    # list / tuple / numpy array
    if isinstance(obj, (list, tuple)):
        arr = np.asarray(obj)
    elif isinstance(obj, np.ndarray):
        arr = obj
    else:
        # 其他 pandas 封装类型，尝试直接转换为 ndarray
        try:
            arr = np.asarray(obj)
        except Exception as e:
            raise ValueError(f'无法将对象转换为图像：{e}')

    if arr.ndim == 1:
        # 可能是扁平化的 uint8 数据，尝试猜测常见分辨率。
        length = arr.size
        # 试一系列常见 (H,W) 配置，优先 480x640, 360x640, 256x256, 128x128
        candidates = [(480,640),(360,640),(256,256),(128,128),(224,224),(720,1280),(600,800)]
        for H,W in candidates:
            if H * W * 3 == length:
                arr2 = arr.reshape((H, W, 3))
                return Image.fromarray(arr2.astype(np.uint8))
        # 退一步：若长度可以被3整除，猜 C,H,W -> try infer sqrt
        if length % 3 == 0:
            n = length // 3
            s = int(np.sqrt(n))
            if s * s == n:
                arr2 = arr.reshape((3, s, s))
                arr2 = np.transpose(arr2, (1,2,0))
                return Image.fromarray(arr2.astype(np.uint8))
        raise ValueError('扁平数组无法推断图像尺寸，请提供 H,W,C 形式或使用其他列。')

    if arr.ndim == 2:
        # 灰度图
        return Image.fromarray(arr.astype(np.uint8)).convert('RGB')

    if arr.ndim == 3:
        # 可能为 (C,H,W) 或 (H,W,C)
        if arr.shape[0] in (1,3,4) and arr.shape[0] != arr.shape[2]:
            # (C,H,W) -> (H,W,C)
            arr2 = np.transpose(arr, (1,2,0))
        else:
            arr2 = arr
        # float -> uint8
        if np.issubdtype(arr2.dtype, np.floating):
            arr2 = (np.clip(arr2, 0.0, 1.0) * 255.0).astype(np.uint8)
        return Image.fromarray(arr2.astype(np.uint8))

    raise ValueError('不支持的数组维度')


def compose_horizontal(images):
    widths, heights = zip(*(img.size for img in images))
    total_w = sum(widths)
    max_h = max(heights)
    new_im = Image.new('RGB', (total_w, max_h))
    x_offset = 0
    for im in images:
        new_im.paste(im, (x_offset, 0))
        x_offset += im.size[0]
    return new_im


def main():
    parser = argparse.ArgumentParser(description='Convert a LeRobot episode parquet to video')
    parser.add_argument('--parquet', required=True, help='输入 parquet 文件路径')
    parser.add_argument('--out', required=True, help='输出视频文件路径 (.mp4 推荐)')
    parser.add_argument('--camera', default='observation.images.cam_high', help='要使用的图像列名（默认 observation.images.cam_high）')
    parser.add_argument('--fps', type=float, default=10.0, help='视频帧率')
    parser.add_argument('--resize', type=int, nargs=2, metavar=('W','H'), help='可选：将每帧缩放到 W H')
    parser.add_argument('--compose', nargs=3, help='可选：传入 3 个列名以横向合成三路相机（如 cam_high cam_left_wrist cam_right_wrist）')
    args = parser.parse_args()

    if not os.path.exists(args.parquet):
        raise SystemExit(f'Parquet 文件不存在: {args.parquet}')

    print(f'读取 parquet (using datasets): {args.parquet} ...')
    hf_ds = datasets.load_dataset('parquet', data_files={'train': [args.parquet]}, split='train')
    print(f'Rows: {len(hf_ds)}  columns: {list(hf_ds.column_names)}')

    # Do not set global transform by default; we'll convert per-row if needed using lerobot helper

    # 检查列
    if args.compose:
        for c in args.compose:
            if c not in hf_ds.column_names:
                raise SystemExit(f'缺少列 {c}，无法 compose')
    else:
        if args.camera not in hf_ds.column_names:
            raise SystemExit(f'缺少列 {args.camera}，请使用 --camera 或 --compose 指定存在的列名')

    # 准备 writer
    out_dir = os.path.dirname(args.out)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    print(f'写入视频: {args.out} （fps={args.fps}）')

    # For imageio v3 (iio) we collect frames and write at the end with iio.imwrite.
    frames_buffer = None
    if _USE_IIO:
        frames_buffer = []
    else:
        writer = imageio.get_writer(args.out, fps=args.fps, codec='libx264', ffmpeg_params=['-pix_fmt', 'yuv420p'])

    try:
        for idx in tqdm(range(len(hf_ds)), desc='frames'):
            row = hf_ds[idx]
            try:
                def _get_pil_from_field(field_value):
                    # If datasets produced PIL.Image instances
                    if isinstance(field_value, Image.Image):
                        return field_value.convert('RGB')
                    # If torch tensors or lists/ndarrays, try conversion via lerobot transform
                    if _HAS_TORCHVISION:
                        try:
                            # Attempt to use hf_transform_to_torch then ToPILImage
                            transformed = hf_transform_to_torch({"tmp": [field_value]})
                            import torch
                            t = transformed["tmp"][0]
                            if isinstance(t, torch.Tensor):
                                pil = ToPILImage()(t)
                                return pil.convert('RGB')
                        except Exception:
                            pass
                    # Fallback: try our generic converter
                    return _to_pil_from_obj(field_value)

                if args.compose:
                    imgs = []
                    for c in args.compose:
                        imgs.append(_get_pil_from_field(row[c]))
                    frame = compose_horizontal(imgs)
                else:
                    frame = _get_pil_from_field(row[args.camera])

                if args.resize:
                    frame = frame.resize((args.resize[0], args.resize[1]), Image.LANCZOS)

                arr = np.asarray(frame)
                if frames_buffer is not None:
                    frames_buffer.append(arr)
                else:
                    writer.append_data(arr)
            except Exception as e:
                print(f'警告：第 {idx} 帧处理失败: {e}，跳过')
                continue
    finally:
        if frames_buffer is not None:
            try:
                # Try writing with ffmpeg plugin
                iio.imwrite(args.out, frames_buffer, fps=args.fps, plugin='ffmpeg')
            except Exception:
                iio.imwrite(args.out, frames_buffer, fps=args.fps)
        else:
            writer.close()

    print('完成。')


if __name__ == '__main__':
    main()
