#from script.eval_policy import usr_args
import torch
import os
import gc
import traceback
import importlib
import sys
import yaml
import numpy as np
import cv2
import threading
sys.path.append("./")
sys.path.append(f"./policy")
sys.path.append("./description/utils")
from envs import CONFIGS_PATH
from envs.utils.create_actor import UnStableError
#import torch.distributed as dist
print('Process', os.getpid(), 'LOCAL_RANK', os.environ.get('LOCAL_RANK'))
os.environ["CUDA_VISIBLE_DEVICES"] = str(os.environ.get('LOCAL_RANK', 0))
_ENV_INIT_LOCK = threading.Lock()
class RobotwinEnvWrapper:
    def __init__(self, usr_args, trial_id, trial_seed, instruction, noise_level):
        self.trial_id = trial_id
        self.trial_seed = trial_seed
        self.usr_args = usr_args
        self.instruction = instruction
        #self.local_rank = local_rank  # 注意这里是 local_rank，不是 gpu_idx
        self.lock = threading.Lock()
        self.env = None
        self.args = None
        self.active = True
        self.complete = False
        self.finish_step = 0
        self.noise_level = noise_level
        self.model = None
    def initialize(self):
        with _ENV_INIT_LOCK:
            with self.lock:
                # 清理GPU缓存
                torch.cuda.empty_cache()
                gc.collect()
                # 重新导入Sapien模块以确保使用新的GPU设置
                import importlib
                import sapien
                importlib.reload(sapien)

                # 现在初始化Sapien环境
                self.env, self.args = get_robotwin2_task(self.usr_args["task_name"], self.usr_args)
                self.env.setup_demo(now_ep_num=self.trial_id, seed=self.trial_seed, is_test=True, **self.args)
                self.env.set_instruction(instruction=self.instruction)
                
                
                # 验证Sapien是否使用了正确的GPU
                if torch.cuda.is_available():
                    memory_allocated = torch.cuda.memory_allocated(0)  # 使用设备0，因为CUDA_VISIBLE_DEVICES已经限制
                    print(f"Process {os.getpid()}: After Sapien init - GPU memory: {memory_allocated/1024**2:.1f}MB")
    
    def get_obs(self):
        """Get observation from environment"""
        with self.lock:
            try:
                got_obs = self.env.get_obs()
                return got_obs
            except Exception as e:
                print(f"****** IN thread: get_obs ERROR {e} ******", flush=True)
                torch.cuda.empty_cache()
                gc.collect()
                got_obs = self.env.get_obs()
                return got_obs
    
    def get_instruction(self):
        """Get instruction for the task"""
        return self.env.get_instruction()
    def get_action(self, obs):
        with self.lock:
            if self.model is None:
                get_model = eval_function_decorator(self.usr_args["policy_name"], "get_model")
                self.usr_args["left_arm_dim"] = len(self.args["left_embodiment_config"]["arm_joints_name"][0])
                self.usr_args["right_arm_dim"] = len(self.args["right_embodiment_config"]["arm_joints_name"][1])
                self.model = get_model(self.usr_args)
            actions, obs_set = self.model.get_action(obs)
            return actions
            #return self.model.get_action(obs)
    def get_action_with_logprob(self, obs):
        with self.lock:
            if self.model is None:
                get_model = eval_function_decorator(self.usr_args["policy_name"], "get_model")
                self.usr_args["left_arm_dim"] = len(self.args["left_embodiment_config"]["arm_joints_name"][0])
                self.usr_args["right_arm_dim"] = len(self.args["right_embodiment_config"]["arm_joints_name"][1])
                self.model = get_model(self.usr_args)
            actions, obs_set, all_trajectory, all_log_prob = self.model.get_action_with_logprob(obs, self.noise_level)
            return actions, obs_set, all_trajectory, all_log_prob
    
    def fm_step(self, actions):
        with self.lock:
            new_obs = None
            try:
                for action in actions:
                    self.env.take_action(action)
                    new_obs = self.env.get_obs()
                    new_obs = encode_obs(new_obs)
                    self.model.update_obs(new_obs)
                    done = self.env.eval_success
                    if done:
                        break
            except Exception as e:
                done = False
                error_msg = f"****** action execution ERROR: {type(e).__name__}: {str(e)} ******"
                print(error_msg, flush=True)
                traceback.print_exc()

            #print(action.shape)
            self.finish_step += len(actions)
            
            if done or self.finish_step >= self.env.step_lim:
                self.active = False
                self.complete = done
            assert new_obs is not None
            obs = new_obs
            
            return obs, done
    def step(self, actions):
        """Execute action in environment"""
        with self.lock:
            try:
                print(actions.shape)
                for action in actions:
                    self.env.take_action(action)
                    done = self.env.eval_success
                    if done:
                        break
            except Exception as e:
                done = False
                error_msg = f"****** action execution ERROR: {type(e).__name__}: {str(e)} ******"
                print(error_msg, flush=True)
                traceback.print_exc()
                
            try:
                obs = self.env.get_obs()
                #obs = encode_obs(obs)
            except Exception as e:
                print(f"****** env.get_obs ERROR {e} ******", flush=True)
                obs = None
            
            #print(action.shape)
            self.finish_step += len(actions)
            
            if done or self.finish_step >= self.env.step_lim:
                self.active = False
                self.complete = done
            
            return obs, done
    def evaluation(self):
        if self.env is None:
            self.initialize()
        
        with self.lock:

            
            
            get_model = eval_function_decorator(self.usr_args["policy_name"], "get_model")
            eval_func = eval_function_decorator(self.usr_args["policy_name"], 'eval')
            reset_func = eval_function_decorator(self.usr_args["policy_name"], 'reset_model')
            self.usr_args["left_arm_dim"] = len(self.args["left_embodiment_config"]["arm_joints_name"][0])
            self.usr_args["right_arm_dim"] = len(self.args["right_embodiment_config"]["arm_joints_name"][1])
            self.model = get_model(self.usr_args)
            print(f"Process {os.getpid()}: Model type: {type(self.model)}")
            
            reset_func(self.model)
            succ = False
            while self.env.take_action_cnt < self.env.step_lim:
                observation = self.env.get_obs()
                eval_func(self.env, self.model, observation)
                if self.env.eval_success:
                    succ = True
                    break
            self.env.close_env(clear_cache=True)
            return succ
    def fm_evaluation(self):
        if self.env is None:
            self.initialize()
        
        with self.lock:
            get_model = eval_function_decorator(self.usr_args["policy_name"], "get_model")
            eval_func = eval_function_decorator(self.usr_args["policy_name"], 'eval')
            reset_func = eval_function_decorator(self.usr_args["policy_name"], 'reset_model')
            self.usr_args["left_arm_dim"] = len(self.args["left_embodiment_config"]["arm_joints_name"][0])
            self.usr_args["right_arm_dim"] = len(self.args["right_embodiment_config"]["arm_joints_name"][1])
            model = get_model(self.usr_args)
            print(f"Process {os.getpid()}: Model type: {type(model)}")
            
            reset_func(model)
            trajectory = []
            while self.env.take_action_cnt < self.env.step_lim:
                observation = self.env.get_obs()
                data = eval_func(self.env, model, observation)
                trajectory.extend(data)
                if self.env.eval_success:
                    self.complete = True
                    break
            self.env.close_env(clear_cache=True)
            return trajectory
            
    def close(self):
        """Close the environment"""
        with self.lock:
            if self.env is not None:
                try:
                    self.env.close_env(clear_cache=True)
                    # 10-14 OOM问题
                    torch.cuda.empty_cache()
                    gc.collect()
                except Exception as e:
                    print(f"******IN env.close ERROR {e} ******", flush=True)

def eval_function_decorator(policy_name, model_name):
    try:
        policy_model = importlib.import_module(policy_name)
        return getattr(policy_model, model_name)
    except ImportError as e:
        raise e
def class_decorator(task_name):
    envs_module = importlib.import_module(f"envs.{task_name}")
    try:
        env_class = getattr(envs_module, task_name)
        env_instance = env_class()
    except:
        raise SystemExit("No Task")
    return env_instance
def get_robotwin2_task(task_name, config):
    """Get robotwin 2.0 task"""
    #robotwin2_path = os.path.join(os.path.dirname(__file__), '..', '..', 'utils', 'envs', 'robotwin2')
    robotwin2_path = os.path.join(os.path.dirname(__file__), '..')
    if robotwin2_path not in sys.path:
        sys.path.append(robotwin2_path)
        
    robotwin2_utils_path = os.path.join(os.path.dirname(__file__), '..', "description", "utils")
    if robotwin2_utils_path not in sys.path:
        sys.path.append(robotwin2_utils_path)
    
    from envs import CONFIGS_PATH
    
    try:
        #env_class = getattr(envs_module, task_name)
        env_instance = class_decorator(task_name)
        #env_instance = env_class()
    except:
        raise SystemExit(f"No Task: {task_name}")
    
    task_config = config.get('task_config', 'demo_clean')
    config_file = os.path.join(robotwin2_path, f"task_config/{task_config}.yml")
    
    with open(config_file, "r", encoding="utf-8") as f:
        args = yaml.load(f.read(), Loader=yaml.FullLoader)
    
    args['task_name'] = task_name
    args['task_config'] = task_config
    args['ckpt_setting'] = config.get('ckpt_setting', 'demo_clean')
    args['expert_data_num'] = config.get('expert_data_num', 50)
    args['instruction_type'] = config.get('instruction_type', 'unseen')
    args['policy_name'] = config.get('policy_name')
    #args['head_camera_type'] = config.get('head_camera_type', 'D435')
    #args['seed'] = config.get('seed', 0)
    #args['checkpoint_num'] = config.get('checkpoint_num', 600)
    

    embodiment_type = args.get("embodiment")
    embodiment_config_path = os.path.join(CONFIGS_PATH, "_embodiment_config.yml")
    
    with open(embodiment_config_path, "r", encoding="utf-8") as f:
        _embodiment_types = yaml.load(f.read(), Loader=yaml.FullLoader)
    
    def get_embodiment_file(embodiment_type):
        robot_file = _embodiment_types[embodiment_type]["file_path"]
        if robot_file is None:
            raise ValueError("No embodiment files")
        return robot_file
    
    def get_embodiment_config(robot_file):
        robot_config_file = os.path.join(robot_file, "config.yml")
        with open(robot_config_file, "r", encoding="utf-8") as f:
            embodiment_args = yaml.load(f.read(), Loader=yaml.FullLoader)
        return embodiment_args
    
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
        raise ValueError("embodiment items should be 1 or 3")
    
    args["left_embodiment_config"] = get_embodiment_config(args["left_robot_file"])
    args["right_embodiment_config"] = get_embodiment_config(args["right_robot_file"])
    
    with open(CONFIGS_PATH + "_camera_config.yml", "r", encoding="utf-8") as f:
        _camera_config = yaml.load(f.read(), Loader=yaml.FullLoader)
    
    head_camera_type = args["camera"]["head_camera_type"]
    args["head_camera_h"] = _camera_config[head_camera_type]["h"]
    args["head_camera_w"] = _camera_config[head_camera_type]["w"]
    
    args["eval_mode"] = True
    args["eval_video_log"] = True
    args["render_freq"] = 0
    args['instruction_type'] = config.get('instruction_type', 'unseen')
    
    return env_instance, args

# def encode_obs(observation):
#     head_cam = cv2.resize(observation["observation"]["head_camera"]["rgb"], (640, 480), interpolation=cv2.INTER_LINEAR)
#     left_cam = cv2.resize(observation["observation"]["left_camera"]["rgb"], (640, 480), interpolation=cv2.INTER_LINEAR)
#     right_cam = cv2.resize(observation["observation"]["right_camera"]["rgb"], (640, 480), interpolation=cv2.INTER_LINEAR)
#     head_cam = np.moveaxis(head_cam, -1, 0) / 255.0
#     left_cam = np.moveaxis(left_cam, -1, 0) / 255.0
#     right_cam = np.moveaxis(right_cam, -1, 0) / 255.0
#     qpos = (observation["joint_action"]["left_arm"] + [observation["joint_action"]["left_gripper"]] +
#             observation["joint_action"]["right_arm"] + [observation["joint_action"]["right_gripper"]])
#     return {
#         "head_cam": head_cam,
#         "left_cam": left_cam,
#         "right_cam": right_cam,
#         "qpos": qpos,
#     }
def encode_obs(observation):
    head_cam = (np.moveaxis(observation["observation"]["head_camera"]["rgb"], -1, 0) / 255)
    left_cam = (np.moveaxis(observation["observation"]["left_camera"]["rgb"], -1, 0) / 255)
    right_cam = (np.moveaxis(observation["observation"]["right_camera"]["rgb"], -1, 0) / 255)
    obs = dict(
        head_cam=head_cam,
        left_cam=left_cam,
        right_cam=right_cam,
    )
    obs["agent_pos"] = observation["joint_action"]["vector"]
    return obs


def test_env(task_name, usr_args, seed, now_id):
    env, args = get_robotwin2_task(task_name, usr_args)
    render_freq = args["render_freq"]
    args["render_freq"] = 0
    args.pop("seed", None)
    try:
        env.setup_demo(now_ep_num=now_id, seed=seed, is_test=True, **args)
        episode_info = env.play_once()
        episode_info_list = [episode_info["info"]]
        from generate_episode_instructions import generate_episode_descriptions
        results = generate_episode_descriptions(task_name, episode_info_list, 100)
        instruction = np.random.choice(results[0][args["instruction_type"]])
        env.close_env()
        args["render_freq"] = render_freq
        return instruction

    except UnStableError as e:
        # print(" -------------")
        # print("Error: ", e)
        # print(" -------------")
        env.close_env()
        return None
    except Exception as e:
        traceback.format_exc()
        # print(" -------------")
        # print("Error: ", e)
        # print(" -------------")
        env.close_env()
        print("error occurs !")
        return None
    