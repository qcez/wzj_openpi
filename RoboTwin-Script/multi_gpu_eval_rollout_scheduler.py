#!/usr/bin/env python
"""
Distribute tasks across GPUs using subprocess.Popen for rollout or eval.

Each task runs in a fully isolated Python process with its own
CUDA_VISIBLE_DEVICES, avoiding Sapien / Vulkan / CUDA context conflicts.

Usage for rollout:
    python multi_gpu_eval_rollout_scheduler.py \
        --eval_mode rollout \
        --config ... \
        --task_names task1 task2 \
        --gpu_num 4 \
        --batch_size_per_gpu 16 \
        --env_seed 100000 200000 \
        --seed_base 100000 \
        --file_name rollout.pkl \
        --overrides ...

Usage for eval:
    python multi_gpu_eval_rollout_scheduler.py \
        --eval_mode eval \
        --config ... \
        --task_names task1 task2 \
        --gpu_num 2 \
        --batch_size 10 \
        --seed_base 0 \
        --overrides ...
"""
import os
import sys
import subprocess
import time
import argparse
import json

TASK_TIMEOUT = 10 * 3600  # 10 hours per task


def get_gpu_ids(num_gpus):
    """Parse physical GPU IDs from Slurm's CUDA_VISIBLE_DEVICES."""
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cvd:
        ids = [g.strip() for g in cvd.split(",") if g.strip()]
    else:
        ids = [str(i) for i in range(num_gpus)]
    return ids[:num_gpus]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_mode", choices=["rollout", "eval"], default="rollout", help="Mode: rollout or eval")
    parser.add_argument("--config", required=True)
    parser.add_argument("--task_names", nargs='+', required=True, help="List of tasks")
    parser.add_argument("--gpu_num", type=int, default=4)
    parser.add_argument("--batch_size_per_gpu", type=int, default=16, help="For rollout")
    parser.add_argument("--batch_size", type=int, default=10, help="For eval, test_num")
    parser.add_argument("--env_seed", nargs='*', type=str, default=[], help="For rollout")
    parser.add_argument("--seed_base", type=int, default=100000, help="For rollout")
    parser.add_argument("--file_name", type=str, default="rollout.pkl", help="For rollout")
    parser.add_argument("--overrides", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    tasks = args.task_names
    gpu_ids = get_gpu_ids(args.gpu_num)
    n_gpus = len(gpu_ids)

    mode = args.eval_mode
    batch_size = args.batch_size_per_gpu if mode == "rollout" else args.batch_size

    if mode == "rollout":
        log_dir = "rollout_logs"
    else:
        log_dir = "eval_logs"  # or something
    os.makedirs(log_dir, exist_ok=True)

    print(f"[Scheduler] Mode: {mode}, {len(tasks)} tasks × batch {batch_size}, GPUs: {gpu_ids}")

    seed_args = []
    if mode == "rollout":
        env_seeds = [int(s) for s in args.env_seed] if args.env_seed else [args.seed_base]
        for s in env_seeds:
            seed_args += ["--env_seed", str(s)]
    override_args = args.overrides or []
    if mode == "eval":
        override_args += ["--seed", str(args.seed_base)]

    # slots[i] = (Popen, task_name, start_time, log_fh) | None
    slots = [None] * n_gpus
    t0 = time.time()

    def reap(idx):
        """Clean up a finished or timed-out slot."""
        if slots[idx] is None:
            return
        proc, name, st, fh = slots[idx]
        if proc.poll() is None:
            proc.kill()
            proc.wait()
        fh.close()
        rc = proc.returncode
        dt = time.time() - st
        print(f"[Scheduler] GPU {gpu_ids[idx]}: '{name}' finished (exit={rc}, {dt:.0f}s)")
        slots[idx] = None

    for task_name in tasks:
        placed = False
        while not placed:
            for i in range(n_gpus):
                if slots[i] is not None:
                    proc = slots[i][0]
                    if proc.poll() is not None:
                        reap(i)
                    elif time.time() - slots[i][2] > TASK_TIMEOUT:
                        print(f"[Scheduler] GPU {gpu_ids[i]}: '{slots[i][1]}' timed out — killing")
                        reap(i)

                if slots[i] is None:
                    gpu = gpu_ids[i]
                    cmd = [
                        sys.executable, "script/multi_gpu_eval_rollout_worker.py",
                        "--config", args.config,
                        "--mode", mode,
                        "--task_name", task_name,
                        "--batch_size", str(batch_size),
                    ]
                    if mode == "rollout":
                        cmd += [
                            "--file_name", args.file_name,
                        ] + seed_args
                    if override_args:
                        cmd += ["--overrides"] + override_args

                    env = os.environ.copy()
                    env["CUDA_VISIBLE_DEVICES"] = gpu
                    env["LOCAL_RANK"] = gpu

                    log_path = os.path.join(log_dir, f"{task_name}_gpu{gpu}_{mode}.log")
                    fh = open(log_path, "w")

                    proc = subprocess.Popen(
                        cmd, env=env,
                        stdout=fh, stderr=subprocess.STDOUT,
                    )
                    slots[i] = (proc, task_name, time.time(), fh)
                    print(f"[Scheduler] GPU {gpu}: '{task_name}' ({mode}) started (pid={proc.pid})")
                    placed = True
                    break

            if not placed:
                time.sleep(10)

    for i in range(n_gpus):
        if slots[i] is not None:
            slots[i][0].wait()
            reap(i)

    dt = time.time() - t0
    print(f"[Scheduler] Wall time: {dt:.0f}s ({dt / 3600:.1f}h)")


if __name__ == "__main__":
    main()