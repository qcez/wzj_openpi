import sys
import os
import subprocess
import copy
import time
import pickle
import json
import multiprocessing
from multiprocessing import Process, Queue
import argparse
import yaml
from datetime import datetime
import importlib
import traceback
from pathlib import Path
import numpy as np
from collections import deque

sys.path.append("./")
sys.path.append(f"./policy")
sys.path.append("./description/utils")
from envs import CONFIGS_PATH
from envs.utils.create_actor import UnStableError
from generate_episode_instructions import *

# Decorators from original scripts
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

def get_camera_config(camera_type):
    camera_config_path = os.path.join(os.path.dirname(__file__), "../task_config/_camera_config.yml")
    assert os.path.isfile(camera_config_path), "task config file is missing"
    with open(camera_config_path, "r", encoding="utf-8") as f:
        args = yaml.load(f.read(), Loader=yaml.FullLoader)
    assert camera_type in args, f"camera {camera_type} is not defined"
    return args[camera_type]

def get_embodiment_config(robot_file):
    robot_config_file = os.path.join(robot_file, "config.yml")
    with open(robot_config_file, "r", encoding="utf-8") as f:
        embodiment_args = yaml.load(f.read(), Loader=yaml.FullLoader)
    return embodiment_args

def parse_args_and_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--generate_rollout", type=str, default="true", help="Generate rollout data instead of evaluation (true/false)")
    parser.add_argument("--task_names", nargs='+', required=True, help="List of tasks to process")
    parser.add_argument("--gpu_num", type=int, default=1, help="Number of GPUs to use")
    parser.add_argument("--batch_size_per_gpu", type=int, default=16, help="Batch size per GPU for rollout")
    parser.add_argument("--file_name", type=str, default="rollout.pkl", help="Output file name for rollout")
    parser.add_argument("--env_seed", nargs='*', type=str, default=[], help="Environment seeds")
    parser.add_argument("--seed_base", type=int, default=100000, help="Base seed")
    parser.add_argument("--use_saved_instructions", type=str, default="false", help="Use saved instructions for evaluation (true/false)")
    parser.add_argument("--saved_instructions_path", type=str, default=None, help="Path to saved instructions JSON")
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

    # Handle env_seed
    if args.env_seed == [] or args.env_seed == ['None'] or args.env_seed == [''] or args.env_seed == ['None']:
        env_seed = None
    else:
        if len(args.env_seed) == 1 and ',' in args.env_seed[0]:
            env_seed = [int(x) for x in args.env_seed[0].split(',') if x.strip()]
        else:
            env_seed = [int(x) for x in args.env_seed]

    config['generate_rollout'] = args.generate_rollout.lower() == 'true'
    config['task_names'] = args.task_names
    config['gpu_num'] = args.gpu_num
    config['batch_size_per_gpu'] = args.batch_size_per_gpu
    config['file_name'] = args.file_name
    config['env_seed'] = env_seed
    config['seed_base'] = args.seed_base
    config['use_saved_instructions'] = args.use_saved_instructions.lower() == 'true'
    config['saved_instructions_path'] = args.saved_instructions_path if args.saved_instructions_path else None

    return config

def worker_rollout(task_queue, gpu_id, config):
    """Worker for rollout generation"""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    print(f"[{datetime.now().strftime('%Y-%m-%d_%H:%M:%S')}]Worker {gpu_id}: Starting rollout worker on GPU {gpu_id}")

    while not task_queue.empty():
        try:
            task_name = task_queue.get_nowait()
        except:
            break

        print(f"[{datetime.now().strftime('%Y-%m-%d_%H:%M:%S')}]Worker {gpu_id}: Processing task {task_name}")

        # Adapted from distributed_loop_collect_fm_wzj.py
        usr_args = config.copy()
        usr_args['task_name'] = task_name

        current_time = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
        batch_size_per_gpu = config['batch_size_per_gpu']
        seed_base = config['seed_base']
        env_seed = config['env_seed']
        gpu_num = config['gpu_num']

        # Simplified rollout logic
        try:
            from script.RobotwinEnvWrapper import RobotwinEnvWrapper, test_env

            task_records = []
            step_cnt = 0
            start_time = time.time()
            while time.time() - start_time < 3600 and len(task_records) < batch_size_per_gpu:
                task_id = gpu_id * batch_size_per_gpu + len(task_records)
                if env_seed is not None:
                    seed = env_seed[gpu_id] + step_cnt * gpu_num * batch_size_per_gpu
                else:
                    seed = seed_base + task_id + 1
                while True:
                    instruction = test_env(task_name, usr_args, seed, task_id)
                    if instruction is not None:
                        break
                    print(f"[{datetime.now().strftime('%Y-%m-%d_%H:%M:%S')}]Worker {gpu_id}: Skipped seed {seed} for task {task_name} (test_env returned None)")
                    # seed += gpu_num * batch_size_per_gpu
                    seed += 1
                    step_cnt += 1
                wrapper = RobotwinEnvWrapper(usr_args, task_id, seed, instruction, 0.1)
                trajectory = wrapper.fm_evaluation()
                episode_length = len(trajectory)
                for step in trajectory:
                    step['episode_length'] = episode_length
                task_record = {
                    'task_id': wrapper.trial_id,
                    'result': wrapper.complete,
                    "trajectory": trajectory,
                    "instruction": wrapper.instruction,  # 添加instruction
                    "seed": None  # rollout中seed不直接对应，设为None
                }
                task_records.append(task_record)
                # Clean up after each rollout
                del wrapper
                import torch
                torch.cuda.empty_cache()
                import gc
                gc.collect()

            # Save results
            data = {
                'time': current_time,
                'results': task_records,
            }
            with open(f"/project/peilab/wzj/RoboTwin/tmp/{current_time}_gpu_{gpu_id}_{task_name}_{config['file_name']}", "wb") as f:
                pickle.dump(data, f)

            # Save JSON with test_instructions format
            success_count = sum(1 for r in task_records if r["result"])
            total_tests = len(task_records)
            success_rate = f"{round(success_count / total_tests * 100, 1)}%" if total_tests > 0 else "0%"
            test_records = [{"test_id": r["task_id"], "seed": r["seed"], "instruction": r["instruction"], "result": "success" if r["result"] else "fail"} for r in task_records]
            rollout_data = {
                "timestamp": current_time,
                "task_name": task_name,
                "policy_name": config.get("policy_name", ""),
                "task_config": config.get("task_config", ""),
                "ckpt_setting": config.get("ckpt_setting", ""),
                "instruction_type": config.get("instruction_type", ""),
                "total_tests": total_tests,
                "success_count": success_count,
                "success_rate": success_rate,
                "replay_mode": False,
                "failed_seeds": [],
                "test_records": test_records
            }
            with open(f"/project/peilab/wzj/RoboTwin/tmp/{current_time}_gpu_{gpu_id}_{task_name}_test_instructions.json", "w", encoding="utf-8") as f:
                json.dump(rollout_data, f, ensure_ascii=False, indent=2)

            print(f"[{datetime.now().strftime('%Y-%m-%d_%H:%M:%S')}]Worker {gpu_id}: Completed rollout for {task_name}")

        except Exception as e:
            print(f"[{datetime.now().strftime('%Y-%m-%d_%H:%M:%S')}]Worker {gpu_id}: Error in task {task_name}: {e}")
            traceback.print_exc()

def worker_eval(task_queue, gpu_id, config):
    """Worker for evaluation"""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    print(f"[{datetime.now().strftime('%Y-%m-%d_%H:%M:%S')}]Worker {gpu_id}: Starting eval worker on GPU {gpu_id}")

    while not task_queue.empty():
        try:
            task_name = task_queue.get_nowait()
        except:
            break

        print(f"[{datetime.now().strftime('%Y-%m-%d_%H:%M:%S')}]Worker {gpu_id}: Processing task {task_name}")

        # Adapted from eval_policy_replay.py
        usr_args = config.copy()
        usr_args["task_name"] = task_name

        try:
            main_eval(usr_args)
            print(f"[{datetime.now().strftime('%Y-%m-%d_%H:%M:%S')}]Worker {gpu_id}: Completed eval for {task_name}")
        except Exception as e:
            print(f"[{datetime.now().strftime('%Y-%m-%d_%H:%M:%S')}]Worker {gpu_id}: Error in task {task_name}: {e}")
            traceback.print_exc()

def main_eval(usr_args):
    """Main evaluation function from eval_policy_replay.py"""
    current_time = datetime.now().strftime("%Y-%m-%d_%H:%M:%S").replace(' ', '_')
    task_name = usr_args["task_name"]
    task_config = usr_args["task_config"]
    ckpt_setting = usr_args["ckpt_setting"]
    policy_name = usr_args["policy_name"]
    instruction_type = usr_args["instruction_type"]

    get_model = eval_function_decorator(policy_name, "get_model")

    with open(f"./task_config/{task_config}.yml", "r", encoding="utf-8") as f:
        args = yaml.load(f.read(), Loader=yaml.FullLoader)

    args['task_name'] = task_name
    args["task_config"] = task_config
    args["ckpt_setting"] = ckpt_setting

    # Embodiment and camera setup (simplified)
    embodiment_type = args.get("embodiment")
    with open(CONFIGS_PATH + "_embodiment_config.yml", "r", encoding="utf-8") as f:
        _embodiment_types = yaml.load(f.read(), Loader=yaml.FullLoader)

    def get_embodiment_file(embodiment_type):
        robot_file = _embodiment_types[embodiment_type]["file_path"]
        if robot_file is None:
            raise "No embodiment files"
        return robot_file

    with open(CONFIGS_PATH + "_camera_config.yml", "r", encoding="utf-8") as f:
        _camera_config = yaml.load(f.read(), Loader=yaml.FullLoader)

    head_camera_type = args["camera"]["head_camera_type"]
    args["head_camera_h"] = _camera_config[head_camera_type]["h"]
    args["head_camera_w"] = _camera_config[head_camera_type]["w"]

    if len(embodiment_type) == 1:
        args["left_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["right_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["dual_arm_embodied"] = True
    elif len(embodiment_type) == 3:
        args["left_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["right_robot_file"] = get_embodiment_file(embodiment_type[1])
        args["embodiment_dis"] = embodiment_type[2]
        args["dual_arm_embodied"] = False
    else:
        raise "embodiment items should be 1 or 3"

    args["left_embodiment_config"] = get_embodiment_config(args["left_robot_file"])
    args["right_embodiment_config"] = get_embodiment_config(args["right_robot_file"])

    save_dir = Path(f"eval_result/{task_name}/{policy_name}/{task_config}/{ckpt_setting}/{current_time}")
    save_dir.mkdir(parents=True, exist_ok=True)

    TASK_ENV = class_decorator(args["task_name"])
    args["policy_name"] = policy_name
    usr_args["left_arm_dim"] = len(args["left_embodiment_config"]["arm_joints_name"][0])
    usr_args["right_arm_dim"] = len(args["right_embodiment_config"]["arm_joints_name"][1])

    # Set video path for evaluation videos (consistent with eval_policy.py)
    TASK_ENV.eval_video_path = str(save_dir)

    video_size = f"{args['head_camera_w']}x{args['head_camera_h']}"

    seed = usr_args["seed"]
    st_seed = 100000 * (1 + seed)
    suc_nums = []
    test_num = 100
    topk = 1

    model = get_model(usr_args)

    use_saved_instructions = usr_args.get("use_saved_instructions", False)
    saved_instructions_path = usr_args.get("saved_instructions_path", None)

    st_seed, suc_num, test_records, replay_info, total_tests = eval_policy(task_name,
                                   TASK_ENV,
                                   args,
                                   model,
                                   st_seed,
                                   test_num=test_num,
                                   video_size=video_size,
                                   instruction_type=instruction_type,
                                   use_saved_instructions=use_saved_instructions,
                                   instructions_json_path=saved_instructions_path)
    suc_nums.append(suc_num)

    file_path = os.path.join(save_dir, f"_result.txt")
    with open(file_path, "w") as file:
        file.write(f"Timestamp: {current_time}\n\n")
        file.write(f"Instruction Type: {instruction_type}\n\n")
        file.write("\n".join(map(str, np.array(suc_nums) / test_num)))

    instructions_json_path = os.path.join(save_dir, "test_instructions.json")
    instructions_data = {
        "timestamp": current_time,
        "task_name": task_name,
        "policy_name": policy_name,
        "task_config": task_config,
        "ckpt_setting": ckpt_setting,
        "instruction_type": instruction_type,
        "total_tests": len(test_records),
        "success_count": suc_num,
        "success_rate": f"{round(suc_num / test_num * 100, 1)}%",
        "replay_mode": replay_info.get("replay_mode", False),
        "failed_seeds": replay_info.get("failed_seeds", []),
        "test_records": test_records
    }

    with open(instructions_json_path, "w", encoding="utf-8") as f:
        json.dump(instructions_data, f, ensure_ascii=False, indent=2)

    print(f"Data saved for {task_name}")

def eval_policy(task_name,
                TASK_ENV,
                args,
                model,
                st_seed,
                test_num=100,
                video_size=None,
                instruction_type=None,
                use_saved_instructions=False,
                instructions_json_path=None):
    print(f"\033[34mTask Name: {args['task_name']}\033[0m")
    print(f"\033[34mPolicy Name: {args['policy_name']}\033[0m")

    # 加载已保存的instruction（如果提供）- 提取seed和instruction的序列
    seed_instruction_pairs = []  # [(seed, instruction), ...]
    if use_saved_instructions and instructions_json_path:
        try:
            with open(instructions_json_path, "r", encoding="utf-8") as f:
                instructions_data = json.load(f)
            # 按顺序提取每条记录的seed和instruction
            for rec in instructions_data["test_records"]:
                seed_instruction_pairs.append((rec["seed"], rec["instruction"]))
            print(f"\033[96m✅ Loaded {len(seed_instruction_pairs)} seed-instruction pairs from {instructions_json_path}\033[0m")
            print(f"\033[96m   将按照JSON中的顺序重放测试 (seed: {seed_instruction_pairs[0][0]}, {seed_instruction_pairs[1][0]}, ...)\033[0m")
        except Exception as e:
            print(f"\033[91m❌ Failed to load saved instructions: {e}\033[0m")
            seed_instruction_pairs = None
    
    # 如果没有加载到seed-instruction对，则按原始模式运行（自动增加seed）
    replay_mode = seed_instruction_pairs is not None and len(seed_instruction_pairs) > 0

    expert_check = True
    TASK_ENV.suc = 0
    TASK_ENV.test_num = 0

    now_id = 0
    succ_seed = 0  # 成功通过Expert Check的测试数
    suc_test_seed_list = []
    failed_seeds = []  # 记录无法通过Expert Check的seed
    
    # 新增：记录所有测试的instruction和结果
    test_records = []

    policy_name = args["policy_name"]
    eval_func = eval_function_decorator(policy_name, "eval")
    reset_func = eval_function_decorator(policy_name, "reset_model")

    now_seed = st_seed
    task_total_reward = 0
    clear_cache_freq = args["clear_cache_freq"]

    args["eval_mode"] = True
    
    # 如果是重放模式，初始化索引
    replay_idx = 0

    while succ_seed < test_num:
        render_freq = args["render_freq"]
        args["render_freq"] = 0

        # 【改动】重放模式 vs 自动模式
        if replay_mode:
            if replay_idx >= len(seed_instruction_pairs):
                break  # 所有JSON中的seed都已测试完
            now_seed, instruction_from_json = seed_instruction_pairs[replay_idx]
        else:
            instruction_from_json = None

        # 【改动】用一个标志来追踪Expert Check是否通过
        expert_check_passed = False
        expert_check_fail_reason = None

        if expert_check:
            try:
                TASK_ENV.setup_demo(now_ep_num=now_id, seed=now_seed, is_test=True, **args)
                episode_info = TASK_ENV.play_once()
                TASK_ENV.close_env()
                expert_check_passed = True
            except UnStableError as e:
                TASK_ENV.close_env()
                expert_check_fail_reason = "UnStableError"
                print(f"\033[93m⚠️  Skip seed {now_seed}: UnStableError (Expert Check failed)\033[0m")
            except Exception as e:
                TASK_ENV.close_env()
                expert_check_fail_reason = f"Exception: {str(e)[:50]}"
                print(f"\033[93m⚠️  Skip seed {now_seed}: Exception (Expert Check failed)\033[0m")
                print("error occurs !")

        if not expert_check_passed:
            # 【改动】Expert Check失败时，也要添加到test_records
            if replay_mode:
                failed_seeds.append((now_seed, expert_check_fail_reason))
                instr_text = seed_instruction_pairs[replay_idx][1]
            else:
                instr_text = "N/A"
            
            current_record = {
                "test_id": TASK_ENV.test_num,
                "seed": now_seed,
                "instruction": instr_text,
                "result": f"Expert Check Failed: {expert_check_fail_reason}"
            }
            test_records.append(current_record)
            
            args["render_freq"] = render_freq
            TASK_ENV.test_num += 1
            
            if replay_mode:
                replay_idx += 1
            else:
                now_seed += 1
            continue

        if (not expert_check) or (TASK_ENV.plan_success and TASK_ENV.check_success()):
            succ_seed += 1
            suc_test_seed_list.append(now_seed)
        else:
            # 【改动】Expert Check通过但check_success返回False时
            expert_check_fail_reason = f"plan_success={TASK_ENV.plan_success}, check_success={TASK_ENV.check_success()}"
            print(f"\033[93m⚠️  Skip seed {now_seed}: Expert Check check failed\033[0m")
            
            if replay_mode:
                failed_seeds.append((now_seed, expert_check_fail_reason))
                instr_text = seed_instruction_pairs[replay_idx][1]
            else:
                instr_text = "N/A"
            
            current_record = {
                "test_id": TASK_ENV.test_num,
                "seed": now_seed,
                "instruction": instr_text,
                "result": f"Expert Check check_success failed: {expert_check_fail_reason}"
            }
            test_records.append(current_record)
            
            args["render_freq"] = render_freq
            TASK_ENV.test_num += 1
            
            if replay_mode:
                replay_idx += 1
            else:
                now_seed += 1
            continue

        args["render_freq"] = render_freq

        TASK_ENV.setup_demo(now_ep_num=now_id, seed=now_seed, is_test=True, **args)
        episode_info_list = [episode_info["info"]]
        results = generate_episode_descriptions(args["task_name"], episode_info_list, test_num)

        # 【改动】优先使用来自JSON的instruction，否则随机生成
        if replay_mode and instruction_from_json is not None:
            instruction = instruction_from_json
            print(f"\033[96m✓ Using instruction from JSON for seed {now_seed}: {instruction[:50]}...\033[0m")
        else:
            instruction = np.random.choice(results[0][instruction_type])

        TASK_ENV.set_instruction(instruction=instruction)  # set language instruction        
        # 记录instruction和相关信息
        current_record = {
            "test_id": TASK_ENV.test_num,
            "seed": now_seed,
            "instruction": instruction
        }
        if TASK_ENV.eval_video_path is not None:
            ffmpeg = subprocess.Popen(
                [
                    "ffmpeg",
                    "-y",
                    "-loglevel",
                    "error",
                    "-f",
                    "rawvideo",
                    "-pixel_format",
                    "rgb24",
                    "-video_size",
                    video_size,
                    "-framerate",
                    "10",
                    "-i",
                    "-",
                    "-pix_fmt",
                    "yuv420p",
                    "-vcodec",
                    "libx264",
                    "-crf",
                    "23",
                    f"{TASK_ENV.eval_video_path}/episode{TASK_ENV.test_num}.mp4",
                ],
                stdin=subprocess.PIPE,
            )
            TASK_ENV._set_eval_video_ffmpeg(ffmpeg)

        succ = False
        reset_func(model)
        while TASK_ENV.take_action_cnt < TASK_ENV.step_lim:
            observation = TASK_ENV.get_obs()
            eval_func(TASK_ENV, model, observation)
            if TASK_ENV.eval_success:
                succ = True
                break
        # task_total_reward += TASK_ENV.episode_score
        if TASK_ENV.eval_video_path is not None:
            TASK_ENV._del_eval_video_ffmpeg()

        if succ:
            TASK_ENV.suc += 1
            print("\033[92mSuccess!\033[0m")
            current_record["result"] = "success"
        else:
            print("\033[91mFail!\033[0m")
            current_record["result"] = "fail"
        
        # 添加测试记录到列表
        test_records.append(current_record)

        now_id += 1
        TASK_ENV.close_env(clear_cache=((succ_seed + 1) % clear_cache_freq == 0))

        if TASK_ENV.render_freq:
            TASK_ENV.viewer.close()

        TASK_ENV.test_num += 1

        print(
            f"\033[93m{task_name}\033[0m | \033[94m{args['policy_name']}\033[0m | \033[92m{args['task_config']}\033[0m | \033[91m{args['ckpt_setting']}\033[0m\n"
            f"Success rate: \033[96m{TASK_ENV.suc}/{TASK_ENV.test_num}\033[0m => \033[95m{round(TASK_ENV.suc/TASK_ENV.test_num*100, 1)}%\033[0m, current seed: \033[90m{now_seed}\033[0m\n"
        )
        # TASK_ENV._take_picture()
        
        # 【改动】根据模式更新seed或索引
        if replay_mode:
            replay_idx += 1
        else:
            now_seed += 1
    
    # 【改动】输出失败seed信息
    if replay_mode and failed_seeds:
        print(f"\n\033[93m━━ 重放过程中跳过的seed ━━\033[0m")
        for failed_seed, reason in failed_seeds:
            print(f"  - seed {failed_seed}: {reason}")
        print(f"\033[96m总计：{len(failed_seeds)} 个seed被跳过 (Expert Check失败)\033[0m\n")
    
    # 准备返回的重放信息
    replay_info = {
        "replay_mode": replay_mode,
        "failed_seeds": failed_seeds
    }
    
    # 返回test_records和重放信息
    return now_seed, TASK_ENV.suc, test_records, replay_info, succ_seed

if __name__ == "__main__":
    multiprocessing.set_start_method('spawn')
    from test_render import Sapien_TEST
    Sapien_TEST()

    config = parse_args_and_config()

    task_names = config['task_names']
    gpu_num = config['gpu_num']
    generate_rollout = config['generate_rollout']

    # Create task queue
    task_queue = Queue()
    for task in task_names:
        task_queue.put(task)

    # Start workers
    processes = []
    for gpu_id in range(gpu_num):
        if generate_rollout:
            p = Process(target=worker_rollout, args=(task_queue, gpu_id, config))
        else:
            p = Process(target=worker_eval, args=(task_queue, gpu_id, config))
        p.start()
        processes.append(p)

    # Wait for all processes to complete
    for p in processes:
        p.join()

    print("All tasks completed.")