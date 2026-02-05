import sys
import os
import subprocess
import copy
import time
import pickle
from gymnasium.experimental.wrappers import PassiveEnvCheckerV0
sys.path.append("./")
sys.path.append(f"./policy")
sys.path.append("./description/utils")
from envs import CONFIGS_PATH
from envs.utils.create_actor import UnStableError

import numpy as np
from pathlib import Path
from collections import deque
import traceback

import yaml
from datetime import datetime
import importlib
import argparse
import pdb

from generate_episode_instructions import *
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import Process, Queue

_ENV_INIT_LOCK = threading.Lock()



current_file_path = os.path.abspath(__file__)
parent_directory = os.path.dirname(current_file_path)


def class_decorator(task_name):
    envs_module = importlib.import_module(f"envs.{task_name}")
    try:
        env_class = getattr(envs_module, task_name)
        env_instance = env_class()
    except:
        raise SystemExit("No Task")
    return env_instance


def eval_function_decorator(policy_name, model_name):
    try:
        policy_model = importlib.import_module(policy_name)
        return getattr(policy_model, model_name)
    except ImportError as e:
        raise e



def parse_args_and_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--file_name", type=str, default="rollout.pkl")
    parser.add_argument("--task_names", nargs='+', required=True)  #多任务
    parser.add_argument("--env_seed", nargs='+', type=int, default=[200004, 200005, 200006, 200007, 200008, 200009, 200010, 200011])
    parser.add_argument("--gpu_num", type=int, default=8)
    parser.add_argument("--batch_size_per_gpu", type=int, default=16)
    parser.add_argument("--overrides", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Parse overrides
    def parse_override_pairs(pairs):
        override_dict = {}
        for i in range(0, len(pairs), 2):
            key = pairs[i].lstrip("--")
            value = pairs[i + 1]
            try:
                value = eval(value)
            except:
                pass
            override_dict[key] = value
        return override_dict

    if args.overrides:
        overrides = parse_override_pairs(args.overrides)
        config.update(overrides)

    config['device'] = f"cuda:{os.environ.get('LOCAL_RANK', 0)}"
    return config, args.file_name, args.task_names, args.env_seed, args.gpu_num, args.batch_size_per_gpu

import subprocess
import json

def run_sapien_distributed_subprocess(usr_args, gpu_num = 1, batch_size_per_gpu = 8, env_seed = None, file_name = None, noise_level = 0.1):
    """使用subprocess让Sapien在不同GPU上运行"""
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    seed_base = 100000
    #instruction = "Pick up the bottle with red screw cap carefully using the right arm"
    print("###########################Start Evaluation###########################")
    start = time.time()
    #file_name = "test3.pkl"
    processes = []
    
    for idx in range(gpu_num):
        gpu_idx = idx
        
        # 创建独立的Python进程，每个进程只看到一个GPU
        cmd = [
            "python", "-c", f"""
import sys
import os
import json
import pickle
from concurrent.futures import ThreadPoolExecutor, as_completed

# 在导入任何模块之前设置环境变量
os.environ["CUDA_VISIBLE_DEVICES"] = "{gpu_idx}"

# 添加路径
sys.path.append('./')
sys.path.append('./policy')
sys.path.append('./description/utils')

import torch
import gc
import traceback
import time

print(f"Process {{os.getpid()}}: CUDA_VISIBLE_DEVICES = {{os.environ.get('CUDA_VISIBLE_DEVICES')}}")
print(f"Process {{os.getpid()}}: CUDA available: {{torch.cuda.is_available()}}")
print(f"Process {{os.getpid()}}: Device count: {{torch.cuda.device_count()}}")

if torch.cuda.is_available():
    print(f"Process {{os.getpid()}}: Current device: {{torch.cuda.current_device()}}")
    print(f"Process {{os.getpid()}}: Device name: {{torch.cuda.get_device_name()}}")

try:
    # 导入Sapien相关模块
    from script.RobotwinEnvWrapper import RobotwinEnvWrapper,encode_obs,test_env,_ENV_INIT_LOCK
    
    batch_results = []
    # 使用正确的参数名称和位置

    wrapper_envs = []
    sys.stdout = open(f"/project/peilab/wzj/RoboTwin/tmp/{current_time}_stdout.txt", "w")
    print(f"Process {{os.getpid()}}: Created stdout file")

    step_cnt = 0
    start_time = time.time()
    while time.time() - start_time < 3600:
        #创建class
        for idx in range({batch_size_per_gpu}):
            task_id = {gpu_idx} * {batch_size_per_gpu} + idx
            if {env_seed} is not None:
                gpu_idx = {gpu_idx}
                env_seed = {env_seed}
                seed = env_seed[gpu_idx] + step_cnt * {gpu_num} * {batch_size_per_gpu}
            else:
                seed = {seed_base} + task_id + 1
            while True:
                instruction = test_env("{usr_args["task_name"]}", {usr_args}, seed, task_id)
                if instruction is not None:
                    break
                seed += {gpu_num} * {batch_size_per_gpu}
                step_cnt += 1
            wrapper = RobotwinEnvWrapper({usr_args}, task_id, seed, instruction, {noise_level})
            wrapper_envs.append(wrapper)
            print(f"Process {{os.getpid()}}: Created environment {{idx}}")

        batch_results = []
        task_records = []
        #进行loop evaluation
        for wrapper in wrapper_envs:
            trajectory = wrapper.fm_evaluation()
            task_record ={{
                'task_id': wrapper.trial_id,
                'result': wrapper.complete,
                "trajectory": trajectory
            }}
            task_records.append(task_record)

        # 输出结果到文件
        with open(f"/project/peilab/wzj/RoboTwin/tmp/{current_time}_gpu_{gpu_idx}_result.pkl", "wb") as f:
            pickle.dump(task_records, f)
        break
        
except Exception as e:
    print(f"Process {{os.getpid()}}: Error in task {{task_id}}: {{e}}")
    traceback.print_exc()
    with open(f"/project/peilab/wzj/RoboTwin/tmp/{current_time}_gpu_{gpu_idx}_error.json", "w") as f:
        json.dump({{"gpu_id": {gpu_idx}, "error": str(e)}}, f)
"""
        ]
        
        print("start process in gpu: ", gpu_idx)
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        processes.append((process, idx, gpu_idx))
    
    # 等待所有进程完成
    results = []
    for process, idx, gpu_idx in processes:
        stdout, stderr = process.communicate()
        print(f"Task {idx} (GPU {gpu_idx}) output:")
        try:
            print(stdout.decode('utf-8'))
        except UnicodeDecodeError:
            print(stdout.decode('utf-8', errors='replace'))
        if stderr:
           print(f"Task {idx} errors:")
           try:
               print(stderr.decode('utf-8'))
           except UnicodeDecodeError:
               print(stderr.decode('utf-8', errors='replace'))
    
    for idx in range(gpu_num):
        gpu_idx = idx
        # 读取结果文件
        try:
            with open(f"/project/peilab/wzj/RoboTwin/tmp/{current_time}_gpu_{gpu_idx}_result.pkl", "rb") as f:
                result_data = pickle.load(f)
                results.extend(result_data)
            #os.remove(f"/project/peilab/wzj/RoboTwin/tmp/gpu_{gpu_idx}_task_{idx}_result.json")
        except Exception as e:
            print(f"Process {os.getpid()}: Error in task {idx}: {e}")
    all_results = [d["result"] for d in results]
    print(f"Total time: {time.time() - start}")
    print(f"Length of results: {len(results)}")
    #print(f"Results: {results}")
    print(f"Success rate: {sum(all_results)/len(all_results)*100:.1f}%")
    
    #with open(f"/project/peilab/wzj/RoboTwin/tmp/{current_time}_overall_result.json", "w") as f:
    if file_name != None:
        with open(f"/project/peilab/wzj/RoboTwin/tmp/{file_name}", "wb") as f:
            data = {
                'time': current_time,
                'results': results,
            }
            pickle.dump(data, f)
    
    return results

if __name__ == "__main__":
    from test_render import Sapien_TEST
    Sapien_TEST()
    usr_args, base_file_name, task_names, env_seed, gpu_num, batch_size_per_gpu = parse_args_and_config()
    print(usr_args)
    print(f"Task names: {task_names}")
    print(f"Env seed: {env_seed}")
    print(f"GPU num: {gpu_num}")
    print(f"Batch size per GPU: {batch_size_per_gpu}")

    # # 临时增加env_seed
    # env_seed = [200004, 200005]
    # env_seed = [200004, 200005, 200006, 200007, 200008, 200009, 200010, 200011]
    
    for task_name in task_names:
        print(f"Processing task: {task_name}")
        usr_args['task_name'] = task_name
        file_name = f"{task_name}_{base_file_name}"
        run_sapien_distributed_subprocess(usr_args, gpu_num, batch_size_per_gpu, env_seed=env_seed, file_name=file_name)
