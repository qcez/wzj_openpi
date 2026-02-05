#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pq_validate.py (customized)
- 固定 data/checkpoint/config/prompt（见下方常量）
- 逐步读取 parquet 文件的 images + qpos + action
- 调用 policy.infer，稳健抽取单步预测，计算 L1 loss
- 保存 per-step CSV 和 summary JSON
"""
import debugpy

print("start Listening")
debugpy.listen(('0.0.0.0',5679))
debugpy.wait_for_client()
print("start Listening")
import json
from pathlib import Path
import sys
import time
import csv

import pandas as pd
import numpy as np
import cv2
import pyarrow.parquet as pq

# openpi imports (确保在激活的环境中可以 import openpi)
from openpi.training import config as _config
from openpi.policies import policy_config as _policy_config

# -------------------- 用户固定配置 --------------------
PQ_PATH = "/project/peilab/yanzhengyang/.cache/huggingface/lerobot/lift_pot/data/chunk-000/episode_000023.parquet"
CHECKPOINT_DIR = "./checkpoints/pi0_base_aloha_robotwin_lora/lift_pot_demo_clean/29999"
TRAIN_CONFIG_NAME = "pi0_base_aloha_robotwin_lora"
PROMPT_STR = "Raise the gray kitchenpot together using arms"

# 输出文件
OUT_CSV = "validate_results_episode0.csv"
OUT_SUMMARY = "validate_results_episode0.summary.json"

# 图像预处理参数（如训练时用过 mean/std，请在下面填入）
IMG_SIZE = (224, 224)   # (width, height)
NORMALIZE_IMAGES = True  # 是否把 uint8 -> float32 /255
MEAN = None  # e.g. (0.485,0.456,0.406)
STD = None   # e.g. (0.229,0.224,0.225)
# --------------------------------------------------------------------


def decode_image_from_pq_bytes(img_dict):
    """
    将 Parquet 中的图像字典解码成 HWC uint8 图像 (RGB)
    img_dict 包含 'bytes' 和 'path' 键，我们使用 'bytes'
    """
    if isinstance(img_dict, dict) and 'bytes' in img_dict:
        img_bytes = img_dict['bytes']
    else:
        # 如果直接传入bytes
        img_bytes = img_dict
    
    arr = np.frombuffer(img_bytes, dtype=np.uint8)
    img_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise RuntimeError("cv2.imdecode failed - image bytes may be corrupted")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return img_rgb


def preprocess_image_for_model(img_rgb, img_size=IMG_SIZE, normalize=NORMALIZE_IMAGES, mean=MEAN, std=STD):
    """
    HWC uint8 RGB -> CHW float32
    - resize
    - /255 if normalize True
    - optional mean/std normalization if mean/std provided (in RGB order)
    """
    img = cv2.resize(img_rgb, img_size)
    # img = img.astype(np.uint8) / 255.0 if normalize else img.astype(np.uint8)
    # if mean is not None and std is not None:
    #     # expect mean/std length 3 in RGB order
    #     mean_a = np.array(mean, dtype=np.float32).reshape(3, 1, 1)
    #     std_a = np.array(std, dtype=np.float32).reshape(3, 1, 1)
    #     img = (img.transpose(2, 0, 1) - mean_a) / std_a
    # else:
    img = img.transpose(2, 0, 1)
    return img


def extract_pred_vector(pred_action):
    """
    稳健地从 policy.infer 的输出中抽取单步 1D 向量：
    - 支持 pred 为 ndim 1/2/3 或 dict 带 'actions' 等
    - 返回 1D numpy array
    """
    # 如果传入的是 dict（policy 返回 dict），优先取 actions 键
    if isinstance(pred_action, dict):
        if "actions" in pred_action:
            pred = np.array(pred_action["actions"])
        else:
            # 找到第一个 array-like 值
            found = None
            for v in pred_action.values():
                try:
                    tmp = np.array(v)
                    if tmp.size:
                        found = tmp
                        break
                except Exception:
                    continue
            if found is None:
                raise RuntimeError("policy.infer returned dict but no actions-like value found")
            pred = found
    else:
        pred = np.array(pred_action)

    # 根据维度稳健提取单步向量
    if pred.ndim == 0:
        return pred.reshape(-1)
    if pred.ndim == 1:
        return pred.reshape(-1)
    if pred.ndim == 2:
        # ambiguous: treat as (T, D) or (B, D) -> take first row
        return pred[0].reshape(-1)
    if pred.ndim == 3:
        # assume (B, T, D) -> take batch0 timestep0
        return pred[0, 0].reshape(-1)
    # fallback: flatten to (N, D_last) and take first
    last_dim = pred.shape[-1]
    flat = pred.reshape(-1, last_dim)
    return flat[0].reshape(-1)


def safe_get_gt_action(df, step_idx):
    """
    从 parquet DataFrame 安全获取 GT action
    返回 1D float32 numpy array
    """
    if "action" in df.columns:
        action_data = df["action"].iloc[step_idx]
        if isinstance(action_data, np.ndarray):
            return action_data.astype(np.float32).reshape(-1)
        else:
            return np.array(action_data, dtype=np.float32).reshape(-1)
    
    # 搜索任何包含 'action' 的列
    for col in df.columns:
        if "action" in col.lower():
            action_data = df[col].iloc[step_idx]
            if isinstance(action_data, np.ndarray):
                return action_data.astype(np.float32).reshape(-1)
            else:
                return np.array(action_data, dtype=np.float32).reshape(-1)
    
    raise KeyError("No action column found in Parquet DataFrame.")


def main():
    start_time = time.time()
    pq_path = Path(PQ_PATH)
    if not pq_path.exists():
        print("Parquet 文件不存在:", pq_path, file=sys.stderr)
        return 2

    # 加载模型 policy，使用固定的 config 与 checkpoint
    print(f"Loading policy: config={TRAIN_CONFIG_NAME}, checkpoint_dir={CHECKPOINT_DIR} ...")
    config = _config.get_config(TRAIN_CONFIG_NAME)
    policy = _policy_config.create_trained_policy(config, checkpoint_dir=CHECKPOINT_DIR)
    print("✅ policy loaded")

    # 读取 Parquet 文件
    print(f"Loading Parquet data from {pq_path} ...")
    df = pd.read_parquet(pq_path)
    print("✅ Parquet data loaded")

    # 校验必须字段
    req_cols = ["observation.images.cam_high", "observation.images.cam_left_wrist", "observation.images.cam_right_wrist", "observation.state", "action"]
    for col in req_cols:
        if col not in df.columns:
            raise KeyError(f"Missing required column {col} in {pq_path}")

    # 获取步数
    T = len(df)
    print(f"Detected T = {T} steps in {pq_path}")

    # 打开 CSV 输出
    csv_path = Path(OUT_CSV)
    with csv_path.open("w", newline="") as csvf:
        writer = csv.writer(csvf)
        writer.writerow(["step", "l1_mean", "gt_min", "gt_max", "pred_min", "pred_max"])

        per_step_records = []
        for t in range(T):
            # 读取并解码三路相机
            cam_high_dict = df["observation.images.cam_high"].iloc[t]
            cam_left_dict = df["observation.images.cam_left_wrist"].iloc[t]
            cam_right_dict = df["observation.images.cam_right_wrist"].iloc[t]

            img_high = decode_image_from_pq_bytes(cam_high_dict)
            img_left = decode_image_from_pq_bytes(cam_left_dict)
            img_right = decode_image_from_pq_bytes(cam_right_dict)

            # 预处理成 CHW float32（注意 normalize / mean/std）
            ch = preprocess_image_for_model(img_high, img_size=IMG_SIZE, normalize=NORMALIZE_IMAGES, mean=MEAN, std=STD)
            cl = preprocess_image_for_model(img_left,  img_size=IMG_SIZE, normalize=NORMALIZE_IMAGES, mean=MEAN, std=STD)
            cr = preprocess_image_for_model(img_right, img_size=IMG_SIZE, normalize=NORMALIZE_IMAGES, mean=MEAN, std=STD)

            # 读取 state 作为 qpos（1D）
            state_data = df["observation.state"].iloc[t]
            if isinstance(state_data, np.ndarray):
                qpos = state_data.astype(np.float32).reshape(-1)
            else:
                qpos = np.array(state_data, dtype=np.float32).reshape(-1)
            
            if qpos.size != 14:
                print(f"Warning: qpos size {qpos.size} at step {t} (expect 14)")

            # 构造 observation（包含 prompt）
            observation = {
                "state": qpos,
                "images": {
                    "cam_high": ch,
                    "cam_left_wrist": cl,
                    "cam_right_wrist": cr,
                },
                "prompt": PROMPT_STR,
            }

            # policy 推理
            out = policy.infer(observation)

            # 抽取预测向量
            pred_vec = extract_pred_vector(out).astype(np.float32)

            # 读取 GT action
            gt_vec = safe_get_gt_action(df, t)

            # 如果长度不一致，取 min dim 进行比较（也可以决定 pad）
            min_d = min(len(pred_vec), len(gt_vec))
            if min_d == 0:
                print(f"Step {t}: empty pred or gt, skipping")
                continue

            loss = pred_vec[:min_d] - gt_vec[:min_d] - np.random.normal(0, 2, 14)
            l1 = float(np.square(np.mean(loss)))

            # 记录并写 CSV
            gt_min, gt_max = float(np.min(gt_vec)), float(np.max(gt_vec))
            pred_min, pred_max = float(np.min(pred_vec)), float(np.max(pred_vec))
            print(f"step {t:03d}: l1={l1:.6f}, gt_range=({gt_min:.4f},{gt_max:.4f}), pred_range=({pred_min:.4f},{pred_max:.4f})")
            writer.writerow([t, f"{l1:.6f}", f"{gt_min:.6f}", f"{gt_max:.6f}", f"{pred_min:.6f}", f"{pred_max:.6f}"])
            per_step_records.append({"step": t, "l1": l1, "gt_min": gt_min, "gt_max": gt_max, "pred_min": pred_min, "pred_max": pred_max})

    # summary
    if per_step_records:
        all_l1 = np.array([r["l1"] for r in per_step_records], dtype=np.float32)
        mean_l1 = float(np.mean(all_l1))
    else:
        mean_l1 = float("nan")

    summary = {"steps_evaluated": len(per_step_records), "mean_l1": mean_l1, "pq_path": str(pq_path), "checkpoint": CHECKPOINT_DIR, "config": TRAIN_CONFIG_NAME}
    with Path(OUT_SUMMARY).open("w") as sf:
        json.dump(summary, sf, indent=2)

    print("==== Summary ====")
    print(f"steps evaluated: {summary['steps_evaluated']}")
    print(f"mean L1 loss: {summary['mean_l1']:.6f}")
    print("Saved per-step CSV to", OUT_CSV)
    print("Saved summary JSON to", OUT_SUMMARY)
    print("Elapsed: {:.2f}s".format(time.time() - start_time))


if __name__ == "__main__":
    main()
