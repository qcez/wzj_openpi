#!/usr/bin/env python
"""
Standalone worker: run rollout or eval for one task on a single GPU.

Must be launched with CUDA_VISIBLE_DEVICES and LOCAL_RANK pre-set in the
process environment (the scheduler / caller handles this).
"""
import sys
import os
import argparse
import traceback
import json
import gc
import importlib
import subprocess
import time

sys.path.append("./")
sys.path.append("./policy")
sys.path.append("./description/utils")

import yaml
import numpy as np


def parse_config(args):
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if args.overrides:
        pairs = args.overrides
        for i in range(0, len(pairs), 2):
            key = pairs[i].lstrip("--")
            val = pairs[i + 1]
            try:
                val = eval(val)
            except Exception:
                pass
            config[key] = val

    config["task_name"] = args.task_name
    config["device"] = "cuda:0"
    return config


def run_rollout(args, config):
    """Rollout logic from gpu_worker.py"""
    task_name = args.task_name

    import torch

    gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "?")
    print(f"[{time.strftime('%Y-%m-%d_%H:%M:%S', time.localtime())}][Worker:{task_name}] GPU={gpu}, CUDA={torch.cuda.is_available()}, devices={torch.cuda.device_count()}")
    if torch.cuda.is_available():
        print(f"[{time.strftime('%Y-%m-%d_%H:%M:%S', time.localtime())}][Worker:{task_name}] {torch.cuda.get_device_name(0)}")

    from script.RobotwinEnvWrapper import RobotwinEnvWrapper, test_env

    results = []
    t0 = time.time()
    shared_model = None

    try:
        for base_seed in args.env_seed:
            seed = base_seed  # Start seed from base_seed
            task_id = 0
            while task_id < args.batch_size:
                while True:
                    instruction = test_env(task_name, config, seed, task_id)
                    if instruction is not None:
                        break
                    seed += 1
                    print(f"[{time.strftime('%Y-%m-%d_%H:%M:%S', time.localtime())}][Worker:{task_name}] seed {seed - 1} unstable, trying {seed}")

                print(f"[{time.strftime('%Y-%m-%d_%H:%M:%S', time.localtime())}][Worker:{task_name}] ep {task_id + 1}/{args.batch_size} seed={seed}")

                try:
                    wrapper = RobotwinEnvWrapper(
                        config, task_id, seed, instruction, args.noise_level,
                    )
                    if shared_model is not None:
                        wrapper.model = shared_model
                    trajectory = wrapper.fm_evaluation()
                    if shared_model is None:
                        shared_model = wrapper.model
                    success = wrapper.complete
                    # Add episode_length to each step
                    episode_length = len(trajectory)
                    for step in trajectory:
                        step['episode_length'] = episode_length
                    results.append({
                        "task_id": task_id, "seed": seed, "result": success, "trajectory": trajectory, "instruction": instruction,
                    })

                    print(f"  => success={success}, frames={len(trajectory)}")

                    del wrapper
                    torch.cuda.empty_cache()
                    gc.collect()

                except Exception as e:
                    print(f"  => ERROR: {e}")
                    traceback.print_exc()
                    results.append({
                        "task_id": task_id, "seed": seed, "result": False, "instruction": instruction,
                    })

                task_id += 1
                seed += 1  # Increment seed for next task
    except Exception as e:
        print(f"Rollout error: {e}")
        traceback.print_exc()

    elapsed = time.time() - t0
    succ_num = sum(1 for r in results if r["result"])
    print(f"\n[{time.strftime('%Y-%m-%d_%H:%M:%S', time.localtime())}][Worker:{task_name}] Done in {elapsed:.0f}s — {succ_num}/{len(results)} success")

    # Output pickle file aligned with distributed_loop_collect_fm_wzj.py
    current_time = time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())
    pickle_data = {
        'time': current_time,
        'results': results,
    }

    # Output rollout_task_instruction.json aligned with eval
    policy_name = config.get("policy_name", "default")
    task_config = config.get("task_config", "default")
    ckpt_setting = config.get("ckpt_setting", "default")
    instruction_type = config.get("instruction_type", "default")

    save_dir = f"rollout_result/{task_name}/{policy_name}/{task_config}/{ckpt_setting}/{current_time}"
    os.makedirs(save_dir, exist_ok=True)

    pickle_file = os.path.join(save_dir, f"{task_name}_{args.file_name}")
    import pickle
    with open(pickle_file, "wb") as f:
        pickle.dump(pickle_data, f)
    print(f"Rollout data saved to {pickle_file}")

    instructions_data = {
        "timestamp": current_time,
        "task_name": task_name,
        "policy_name": policy_name,
        "task_config": task_config,
        "ckpt_setting": ckpt_setting,
        "instruction_type": instruction_type,
        "total_tests": len(results),
        "success_count": succ_num,
        "success_rate": f"{round(succ_num / len(results) * 100, 1)}%" if results else "0%",
        "elapsed_time": elapsed,
        "test_records": [{"seed": r["seed"], "success": r["result"], "instruction": r["instruction"], "task_id": r["task_id"]} for r in results]
    }

    instructions_json_path = os.path.join(save_dir, "rollout_task_instruction.json")
    with open(instructions_json_path, "w", encoding="utf-8") as f:
        json.dump(instructions_data, f, ensure_ascii=False, indent=2)

    print(f"Rollout instructions saved to {instructions_json_path}")


def run_eval(args, config):
    """Eval logic aligned with eval_policy.py"""
    import importlib
    from envs import CONFIGS_PATH
    from envs.utils.create_actor import UnStableError
    from generate_episode_instructions import generate_episode_descriptions

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

    current_time =  time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())
    task_name = config["task_name"]
    task_config = config.get("task_config", "default")
    ckpt_setting = config.get("ckpt_setting", "default")
    policy_name = config.get("policy_name", "default")
    instruction_type = config.get("instruction_type", "default")

    with open(f"./task_config/{task_config}.yml", "r", encoding="utf-8") as f:
        task_args = yaml.safe_load(f.read())

    task_args['task_name'] = task_name
    task_args["task_config"] = task_config
    task_args["ckpt_setting"] = ckpt_setting

    # Embodiment and camera setup
    embodiment_type = task_args.get("embodiment")
    with open(CONFIGS_PATH + "_embodiment_config.yml", "r", encoding="utf-8") as f:
        _embodiment_types = yaml.safe_load(f.read())

    def get_embodiment_file(embodiment_type):
        robot_file = _embodiment_types[embodiment_type]["file_path"]
        if robot_file is None:
            raise "No embodiment files"
        return robot_file

    with open(CONFIGS_PATH + "_camera_config.yml", "r", encoding="utf-8") as f:
        _camera_config = yaml.safe_load(f.read())

    head_camera_type = task_args["camera"]["head_camera_type"]
    task_args["head_camera_h"] = _camera_config[head_camera_type]["h"]
    task_args["head_camera_w"] = _camera_config[head_camera_type]["w"]

    if len(embodiment_type) == 1:
        task_args["left_robot_file"] = get_embodiment_file(embodiment_type[0])
        task_args["right_robot_file"] = get_embodiment_file(embodiment_type[0])
        task_args["dual_arm_embodied"] = True
    elif len(embodiment_type) == 3:
        task_args["left_robot_file"] = get_embodiment_file(embodiment_type[0])
        task_args["right_robot_file"] = get_embodiment_file(embodiment_type[1])
        task_args["embodiment_dis"] = embodiment_type[2]
        task_args["dual_arm_embodied"] = False
    else:
        raise "embodiment items should be 1 or 3"

    task_args["left_embodiment_config"] = yaml.safe_load(open(task_args["left_robot_file"] + "/config.yml").read())
    task_args["right_embodiment_config"] = yaml.safe_load(open(task_args["right_robot_file"] + "/config.yml").read())

    save_dir = f"eval_result/{task_name}/{policy_name}/{task_config}/{ckpt_setting}/{current_time}"
    os.makedirs(save_dir, exist_ok=True)

    TASK_ENV = class_decorator(task_args["task_name"])
    task_args["policy_name"] = policy_name
    config["left_arm_dim"] = len(task_args["left_embodiment_config"]["arm_joints_name"][0])
    config["right_arm_dim"] = len(task_args["right_embodiment_config"]["arm_joints_name"][1])

    TASK_ENV.eval_video_path = save_dir
    video_size = f"{task_args['head_camera_w']}x{task_args['head_camera_h']}"

    seed = config.get("seed", 0)
    st_seed = 100000 * (1 + seed)
    test_num = args.batch_size

    model = eval_function_decorator(policy_name, "get_model")(config)

    print(f"Task Name: {task_args['task_name']}")
    print(f"Policy Name: {task_args['policy_name']}")

    expert_check = True
    TASK_ENV.suc = 0
    TASK_ENV.test_num = 0

    now_id = 0
    succ_seed = 0
    test_records = []

    policy_name = task_args["policy_name"]
    eval_func = eval_function_decorator(policy_name, "eval")
    reset_func = eval_function_decorator(policy_name, "reset_model")

    now_seed = st_seed
    clear_cache_freq = task_args.get("clear_cache_freq", 10)

    task_args["eval_mode"] = True

    t0 = time.time()
    while succ_seed < test_num:
        render_freq = task_args["render_freq"]
        task_args["render_freq"] = 0

        if expert_check:
            try:
                TASK_ENV.setup_demo(now_ep_num=now_id, seed=now_seed, is_test=True, **task_args)
                episode_info = TASK_ENV.play_once()
                TASK_ENV.close_env()
            except UnStableError as e:
                TASK_ENV.close_env()
                now_seed += 1
                task_args["render_freq"] = render_freq
                continue
            except Exception as e:
                TASK_ENV.close_env()
                now_seed += 1
                task_args["render_freq"] = render_freq
                print("error occurs !")
                continue

        if (not expert_check) or (TASK_ENV.plan_success and TASK_ENV.check_success()):
            succ_seed += 1
        else:
            now_seed += 1
            task_args["render_freq"] = render_freq
            continue

        task_args["render_freq"] = render_freq

        TASK_ENV.setup_demo(now_ep_num=now_id, seed=now_seed, is_test=True, **task_args)
        episode_info_list = [episode_info["info"]]
        results = generate_episode_descriptions(task_args["task_name"], episode_info_list, test_num)

        instruction = np.random.choice(results[0][instruction_type]) if results[0][instruction_type] else "default"

        TASK_ENV.set_instruction(instruction=instruction)

        current_record = {
            "test_id": TASK_ENV.test_num,
            "seed": now_seed,
            "instruction": instruction
        }

        succ = False
        reset_func(model)
        while TASK_ENV.take_action_cnt < TASK_ENV.step_lim:
            observation = TASK_ENV.get_obs()
            eval_func(TASK_ENV, model, observation)
            if TASK_ENV.eval_success:
                succ = True
                break

        if succ:
            TASK_ENV.suc += 1
            print("\033[92mSuccess!\033[0m")
            current_record["result"] = "success"
        else:
            print("\033[91mFail!\033[0m")
            current_record["result"] = "fail"

        test_records.append(current_record)

        now_id += 1
        TASK_ENV.close_env(clear_cache=((succ_seed + 1) % clear_cache_freq == 0))

        TASK_ENV.test_num += 1

        print(
            f"\033[93m{task_name}\033[0m | \033[94m{task_args['policy_name']}\033[0m | \033[92m{task_args['task_config']}\033[0m | \033[91m{task_args['ckpt_setting']}\033[0m\n"
            f"Success rate: \033[96m{TASK_ENV.suc}/{TASK_ENV.test_num}\033[0m => \033[95m{round(TASK_ENV.suc/TASK_ENV.test_num*100, 1)}%\033[0m, current seed: \033[90m{now_seed}\033[0m\n"
        )
        now_seed += 1

    elapsed = time.time() - t0
    succ_num = sum(1 for r in test_records if r["result"] == "success")
    print(f"\n[{time.strftime('%Y-%m-%d_%H:%M:%S', time.localtime())}][Worker:{task_name}] Eval done in {elapsed:.0f}s — {succ_num}/{len(test_records)} success")

    instructions_data = {
        "timestamp": current_time,
        "task_name": task_name,
        "policy_name": policy_name,
        "task_config": task_config,
        "ckpt_setting": ckpt_setting,
        "instruction_type": instruction_type,
        "total_tests": len(test_records),
        "success_count": succ_num,
        "success_rate": f"{round(succ_num / len(test_records) * 100, 1)}%" if test_records else "0%",
        "elapsed_time": elapsed,
        "test_records": test_records
    }

    instructions_json_path = os.path.join(save_dir, f"{current_time}_task_instruction.json")
    with open(instructions_json_path, "w", encoding="utf-8") as f:
        json.dump(instructions_data, f, ensure_ascii=False, indent=2)

    print(f"Test instructions saved to {instructions_json_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--mode", choices=["rollout", "eval"], default="rollout")
    parser.add_argument("--task_name", required=True)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--env_seed", type=int, nargs="+", default=[200000])
    parser.add_argument("--noise_level", type=float, default=0.1)
    parser.add_argument("--file_name", type=str, default="rollout.pkl")
    parser.add_argument("--overrides", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    config = parse_config(args)
    if args.mode == "rollout":
        run_rollout(args, config)
    else:
        run_eval(args, config)

    os._exit(0)


if __name__ == "__main__":
    main()