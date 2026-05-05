#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
import time
import signal
import numpy as np
import argparse
from collections import deque, defaultdict
import threading

from openpi_client import image_tools
from openpi_client import websocket_client_policy
from pyAgxArm import create_agx_arm_config, AgxArmFactory

import rospy
from std_msgs.msg import Header
from geometry_msgs.msg import Twist
from sensor_msgs.msg import JointState, Image
from nav_msgs.msg import Odometry
from cv_bridge import CvBridge

import h5py
import os
import termios
import select
import tty

# ==================== 全局计数器 (For tracking number of rollout data) ====================
episode_idx_dict = {"success": 0, "fail": 0}
success_count = 0
total_count = 0

#==================== 异步推理全局变量(Global variables for asynchronous inference) ==========================
_action_prod_thread = None
_action_stop_event = threading.Event()
_action_lock = threading.Lock()

_ACTION_CHUNK_SIZE_ORIGIN = 50  # 模型一次推理的步数(The number of steps for one model inference)
_ACTION_DIM = 14

_action_chunks = {}
_control_t = 0

# ========== 线程停止信号 (For notifying the producer thread) ===============
stop_signal = threading.Event() 

#====================== 异步推理函数(For Temporal Ensembling) ============================
def clear_action_buffer(): # 清空缓存 (Clear cache)
    global _control_t, _action_chunks
    with _action_lock:
        _control_t = 0
        _action_chunks = {} # {cursor: chunk} 

def temporal_ensembled_action(current_time):
    """
    取当前时刻所有已推断的动作softmax加权平均 
    (Take the softmax weighted average of all inferred actions at the current moment)
    """
    with _action_lock:
        relevant = {}
        to_delete = []
        before_keys = sorted(_action_chunks.keys())

        for cursor, chunk in _action_chunks.items():
            end = cursor + _ACTION_CHUNK_SIZE_ORIGIN # 模型一次推理的步数
            if cursor <= current_time < end:
                # 还覆盖当前时刻 → 参与加权 (Still covers the current moment → participates in weighting)
                relevant[cursor] = chunk[current_time - cursor]
            elif end <= current_time:
                # 整个chunk都过期了 → 标记删除 (The entire chunk has expired → mark for deletion)
                to_delete.append(cursor)

        # 清理过期chunk，并打印关键状态用于排查是否正确删除 (Clean up expired chunks and print key status to troubleshoot if deletion is correct)
        print(
            f"[action_chunks] t={current_time} before={before_keys} "
            f"delete={sorted(to_delete)}"
        )
        for k in to_delete:
            del _action_chunks[k]
        print(f"[action_chunks] t={current_time} after={sorted(_action_chunks.keys())}")
        
        if not relevant: 
            return None

        # 旧的权重低，新的权重高 (older -> lower weight, newer -> higher weight)
        sorted_items = sorted(relevant.items(), key=lambda x: x[0])
        cur_actions = np.asarray([action for _, action in sorted_items])

    exp_weights = np.exp(0.3 * np.arange(len(cur_actions)))
    exp_weights = (exp_weights / exp_weights.sum())[:, None]
    sub_action = (cur_actions * exp_weights).sum(axis=0)
    return sub_action

def _enqueue_chunk_to_expected_queues(chunk, cursor):
    """
    将一个 chunk 存入字典 
    (Store a chunk in the dictionary)
    """
    with _action_lock:
        _action_chunks[cursor] = chunk

#====================== hdf5保存函数(For saving hdf5 file) ============================
def save_data(images, actions, states, success, folder_type, args):
    """
    保存数据到 HDF5 文件
    (Save data to HDF5 file)
    """
    print("final长度：",len(images), len(actions), len(states))
    global episode_idx_dict 
    data_size = len(actions)
    if data_size == 0:
        print("没有数据可保存(There is no data to save)")
        return
    
    # 数据结构初始化 (Data structure initialization)
    data_dict = {
        '/observations/qpos': [],
        '/observations/qvel': [], # fill with zeros
        '/observations/effort': [], # fill with zeros
        '/action': [],
        '/base_action': [], # fill with zeros
        '/observations/images/cam_high': [],
        '/observations/images/cam_left_wrist': [],
        '/observations/images/cam_right_wrist': [],
        '/isSuccess': []
    }
    
    bridge = CvBridge()
    
    # 处理图像和数据对齐 (Process images and align data)
    for i, action in enumerate(actions):
        if i < len(images):
            img_tuple = images[i]

            img_high = bridge.imgmsg_to_cv2(img_tuple[0], 'passthrough')
            img_left = bridge.imgmsg_to_cv2(img_tuple[1], 'passthrough')
            img_right = bridge.imgmsg_to_cv2(img_tuple[2], 'passthrough')
            
            data_dict['/observations/images/cam_high'].append(img_high)
            data_dict['/observations/images/cam_left_wrist'].append(img_left)
            data_dict['/observations/images/cam_right_wrist'].append(img_right)
        
        if i < len(states):
            data_dict['/observations/qpos'].append(states[i])
            # qvel/effort fill with zeros (pyAgxArm does not support reading this information))
            data_dict['/observations/qvel'].append(np.zeros(14))
            data_dict['/observations/effort'].append(np.zeros(14))

        data_dict['/action'].append(action)
        data_dict['/isSuccess'].append(success)
        data_dict['/base_action'].append([0.0, 0.0])
    
    print("data_dict['/observations/qpos']:" , len(data_dict['/observations/qpos']))

    if folder_type not in episode_idx_dict:
        episode_idx_dict[folder_type] = 0
    idx = episode_idx_dict[folder_type]
    base_path = os.path.join(args.output_dir, folder_type)
    if not os.path.exists(base_path):
        os.makedirs(base_path, exist_ok=True)
    
    dataset_path = os.path.join(base_path, f"episode_{idx}.hdf5")
    
    # 写入 HDF5 (Write to HDF5)
    t0 = time.time()
    with h5py.File(dataset_path, 'w', rdcc_nbytes=1024**2*2) as root:
        root.attrs['sim'] = False
        root.attrs['compress'] = False
        
        obs = root.create_group('observations')
        image = obs.create_group('images')
        
        # 根据实际图像大小调整 shape，480x640x3 (Adjust shape according to actual image size, 480x640x3)
        h, w, c = data_dict['/observations/images/cam_high'][0].shape
        
        _ = image.create_dataset("cam_high", (data_size, h, w, c), dtype='uint8', chunks=(1, h, w, c))
        _ = image.create_dataset("cam_left_wrist", (data_size, h, w, c), dtype='uint8', chunks=(1, h, w, c))
        _ = image.create_dataset("cam_right_wrist", (data_size, h, w, c), dtype='uint8', chunks=(1, h, w, c))
        
        _ = obs.create_dataset('qpos', (data_size, 14))
        _ = obs.create_dataset('qvel', (data_size, 14))
        _ = obs.create_dataset('effort', (data_size, 14))
        _ = root.create_dataset('base_action', (data_size, 2))
        _ = root.create_dataset('action', (data_size, 14))
        _ = root.create_dataset('isSuccess', (data_size))

        for name, array in data_dict.items():
            root[name][...] = array
    
    print(f'\033[32m Saving: {time.time() - t0:.1f} secs. {dataset_path} \033[0m')
    episode_idx_dict[folder_type] += 1
#====================== 用于获取图像(For getting images) ============================
class RosOperator:
    def __init__(self, args):
        self.img_front_deque = None
        self.img_front = None
        self.img_left = None
        self.img_right = None
        self.img_right_deque = None
        self.img_left_deque = None
        self.img_front_depth_deque = None
        self.img_right_depth_deque = None
        self.img_left_depth_deque = None
        self.bridge = None
        self.args = args
        self.ctrl_state = False
        self.ctrl_state_lock = threading.Lock()
        self.init()
        self.init_ros()

    def init(self):
        self.bridge = CvBridge()
        self.img_left_deque = deque()
        self.img_right_deque = deque()
        self.img_front_deque = deque()
        self.img_left_depth_deque = deque()
        self.img_right_depth_deque = deque()
        self.img_front_depth_deque = deque()
    
    def get_img(self):
        
        if len(self.img_front_deque) == 0 :
            return False
        frame_time = self.img_front_deque[-1].header.stamp.to_sec()

        if len(self.img_front_deque) == 0 < frame_time:
            return False
        while self.img_front_deque[0].header.stamp.to_sec() < frame_time:
            self.img_front_deque.popleft()
        img_front = self.bridge.imgmsg_to_cv2(self.img_front_deque.popleft(), 'passthrough')
        return img_front
    def get_frame(self):
        # if len(self.img_right_deque) == 0 or len(self.img_front_deque) == 0 or \
        #         (self.args.use_depth_image and (len(self.img_left_depth_deque) == 0 or len(self.img_right_depth_deque) == 0 or len(self.img_front_depth_deque) == 0)):
        #     return False
        if len(self.img_right_deque) == 0 or len(self.img_front_deque) == 0 or len(self.img_left_deque) == 0 or \
            (self.args.use_depth_image and (len(self.img_left_depth_deque) == 0 or len(self.img_right_depth_deque) == 0 or len(self.img_front_depth_deque) == 0)):
            return False
        if self.args.use_depth_image:
            frame_time = min([self.img_left_deque[-1].header.stamp.to_sec(), self.img_right_deque[-1].header.stamp.to_sec(), self.img_front_deque[-1].header.stamp.to_sec(),
                              self.img_left_depth_deque[-1].header.stamp.to_sec(), self.img_right_depth_deque[-1].header.stamp.to_sec(), self.img_front_depth_deque[-1].header.stamp.to_sec()])
        else:
            frame_time = min([self.img_left_deque[-1].header.stamp.to_sec(),self.img_right_deque[-1].header.stamp.to_sec(), self.img_front_deque[-1].header.stamp.to_sec()])

        if len(self.img_left_deque) == 0 or self.img_left_deque[-1].header.stamp.to_sec() < frame_time:
            return False
        if len(self.img_right_deque) == 0 or self.img_right_deque[-1].header.stamp.to_sec() < frame_time:
            return False
        if len(self.img_front_deque) == 0 or self.img_front_deque[-1].header.stamp.to_sec() < frame_time:
            return False
        if self.args.use_depth_image and (len(self.img_left_depth_deque) == 0 or self.img_left_depth_deque[-1].header.stamp.to_sec() < frame_time):
            return False
        if self.args.use_depth_image and (len(self.img_right_depth_deque) == 0 or self.img_right_depth_deque[-1].header.stamp.to_sec() < frame_time):
            return False
        if self.args.use_depth_image and (len(self.img_front_depth_deque) == 0 or self.img_front_depth_deque[-1].header.stamp.to_sec() < frame_time):
            return False


        while self.img_left_deque[0].header.stamp.to_sec() < frame_time:
            self.img_left_deque.popleft()
        img_left = self.bridge.imgmsg_to_cv2(self.img_left_deque.popleft(), 'passthrough')

        while self.img_right_deque[0].header.stamp.to_sec() < frame_time:
            self.img_right_deque.popleft()
        img_right = self.bridge.imgmsg_to_cv2(self.img_right_deque.popleft(), 'passthrough')

        while self.img_front_deque[0].header.stamp.to_sec() < frame_time:
            self.img_front_deque.popleft()
        img_front = self.bridge.imgmsg_to_cv2(self.img_front_deque.popleft(), 'passthrough')

        img_left_depth = None
        if self.args.use_depth_image:
            while self.img_left_depth_deque[0].header.stamp.to_sec() < frame_time:
                self.img_left_depth_deque.popleft()
            img_left_depth = self.bridge.imgmsg_to_cv2(self.img_left_depth_deque.popleft(), 'passthrough')

        img_right_depth = None
        if self.args.use_depth_image:
            while self.img_right_depth_deque[0].header.stamp.to_sec() < frame_time:
                self.img_right_depth_deque.popleft()
            img_right_depth = self.bridge.imgmsg_to_cv2(self.img_right_depth_deque.popleft(), 'passthrough')

        img_front_depth = None
        if self.args.use_depth_image:
            while self.img_front_depth_deque[0].header.stamp.to_sec() < frame_time:
                self.img_front_depth_deque.popleft()
            img_front_depth = self.bridge.imgmsg_to_cv2(self.img_front_depth_deque.popleft(), 'passthrough')


        return (img_front, img_left, img_right, img_front_depth, img_left_depth, img_right_depth)

    def img_left_callback(self, msg):
        if len(self.img_left_deque) >= 2000:
            self.img_left_deque.popleft()
        self.img_left_deque.append(msg)
        self.img_left = msg

    def img_right_callback(self, msg):
        if len(self.img_right_deque) >= 2000:
            self.img_right_deque.popleft()
        self.img_right_deque.append(msg)
        self.img_right = msg

    def img_front_callback(self, msg):
        if len(self.img_front_deque) >= 2000:
            self.img_front_deque.popleft()
        self.img_front_deque.append(msg)
        self.img_front = msg

    def img_left_depth_callback(self, msg):
        if len(self.img_left_depth_deque) >= 2000:
            self.img_left_depth_deque.popleft()
        self.img_left_depth_deque.append(msg)

    def img_right_depth_callback(self, msg):
        if len(self.img_right_depth_deque) >= 2000:
            self.img_right_depth_deque.popleft()
        self.img_right_depth_deque.append(msg)

    def img_front_depth_callback(self, msg):
        if len(self.img_front_depth_deque) >= 2000:
            self.img_front_depth_deque.popleft()
        self.img_front_depth_deque.append(msg)

    def ctrl_callback(self, msg):
        self.ctrl_state_lock.acquire()
        self.ctrl_state = msg.data
        self.ctrl_state_lock.release()

    def get_ctrl_state(self):
        self.ctrl_state_lock.acquire()
        state = self.ctrl_state
        self.ctrl_state_lock.release()
        return state

    def init_ros(self):
        rospy.init_node('joint_state_publisher', anonymous=True)
        rospy.Subscriber(self.args.img_front_topic, Image, self.img_front_callback, queue_size=1000, tcp_nodelay=True)
        rospy.Subscriber(self.args.img_left_topic, Image, self.img_left_callback, queue_size=1000, tcp_nodelay=True)
        rospy.Subscriber(self.args.img_right_topic, Image, self.img_right_callback, queue_size=1000, tcp_nodelay=True)
        if self.args.use_depth_image:
            rospy.Subscriber(self.args.img_left_depth_topic, Image, self.img_left_depth_callback, queue_size=1000, tcp_nodelay=True)
            rospy.Subscriber(self.args.img_right_depth_topic, Image, self.img_right_depth_callback, queue_size=1000, tcp_nodelay=True)
            rospy.Subscriber(self.args.img_front_depth_topic, Image, self.img_front_depth_callback, queue_size=1000, tcp_nodelay=True)


#======================= 推理主程序(Main procedure for inference) =============================
class InferController:
    def __init__(self, host="localhost", port=8000):
        self.client = None
        self.left_channel = "can_left"
        self.right_channel = "can_right"
        self.bitrate = 1000000
        
        # === 机械臂参数 (Robot arm parameters) ===
        self.speed_pct = 15           # 速度百分比(velocity percentage)
        self.max_linear_vel = 0.5     # m/s
        self.max_angular_vel = 0.1    # rad/s
        self.max_linear_acc = 0.1     # m/s2
        self.max_angular_acc = 0.05   # rad/s2

        # === 夹爪对象 (Gripper objects) ===
        self.left_gripper = None
        self.right_gripper = None
        
        # === 指令 (Instructions) ===
        self.instruction = "fold the cloth"
        
        # === 执行参数 (Execution parameters) ===
        self.ACTION_CHUNK_SIZE = 50   # 执行前self.ACTION_CHUNK_SIZE步() (Execute the first self.ACTION_CHUNK_SIZE steps before checking for new instructions)
        
        # === 初始位置 (Initial position) ===
        # folding_cloth
        self.LEFT_INIT_POSITION = [-0.017, 1.293, -1.166, -0.323, 1.034, 0.486, 0.058]
        self.RIGHT_INIT_POSITION = [0.019, 1.565, -1.226, 0.4, 1.029, -0.258, 0.071]     
        
        # === 安全阈值 (Safety thresholds) ===
        self.ACTION_SAFETY_THRESHOLD = 1.5
        self.STATE_SAFETY_THRESHOLD = 0.3

    def connect_arms(self):
        """
        连接双臂并初始化夹爪 
        (Connect dual arms and initialize grippers)
        """
        print("连接双臂(connect arms)...")
        try:
            # --- 左臂 (Left arm) ---
            cfg_l = create_agx_arm_config(
                robot="piper", comm="can", channel=self.left_channel, bitrate=self.bitrate
            )
            self.left_arm = AgxArmFactory.create_arm(cfg_l)
            # 初始化夹爪 (Initialize gripper)
            self.left_gripper = self.left_arm.init_effector(
                self.left_arm.OPTIONS.EFFECTOR.AGX_GRIPPER
            )
            self.left_arm.connect()
            
            # --- 右臂 (Right arm) ---
            cfg_r = create_agx_arm_config(
                robot="piper", comm="can", channel=self.right_channel, bitrate=self.bitrate
            )
            self.right_arm = AgxArmFactory.create_arm(cfg_r)
            # 初始化夹爪 (Initialize gripper)
            self.right_gripper = self.right_arm.init_effector(
                self.right_arm.OPTIONS.EFFECTOR.AGX_GRIPPER
            )
            self.right_arm.connect()
            
            time.sleep(0.5)
            if not (self.left_arm.is_ok() and self.right_arm.is_ok()):
                raise Exception("机械臂连接状态检查失败(connection status check failed)")
            
            # 设置速度和使能 (Set speed and enable)
            for name, arm in [("左臂left arm", self.left_arm), ("右臂right arm", self.right_arm)]:
                arm.set_flange_vel_acc_limits(
                    max_linear_vel=self.max_linear_vel,
                    max_angular_vel=self.max_angular_vel,
                    max_linear_acc=self.max_linear_acc,
                    max_angular_acc=self.max_angular_acc,
                    timeout=1.0
                )
                arm.set_speed_percent(self.speed_pct)
                
                enabled = False
                for _ in range(5):
                    if arm.enable():
                        enabled = True
                        break
                    time.sleep(0.5)
                if not enabled:
                    raise Exception(f"{name} 使能超时(enable timeout)")
                
            print("连接成功(connect successfully)\n")
            return True
            
        except Exception as e:
            print(f"连接错误：{e}")
            return False
    
    def get_status_and_state(self):
        """
        获取当前状态：12 维关节 (6+6) + 2 维夹爪 (归一化) = 14 维 
        (Get current state: 12-dimensional joints (6+6) + 2-dimensional grippers (normalized) = 14 dimensions)
        """
        state = np.zeros(14, dtype=np.float32)
        try:
            # 1. 左臂关节 (1. Left arm joints)
            ja_l = self.left_arm.get_joint_angles()
            if ja_l is not None: 
                state[0:6] = ja_l.msg
            
            # 2. 右臂关节 (2. Right arm joints)
            ja_r = self.right_arm.get_joint_angles()
            if ja_r is not None: 
                state[7:13] = ja_r.msg
            
            # 3. 左臂夹爪 (3. Left arm gripper)
            if self.left_gripper:
                gs = self.left_gripper.get_gripper_status()
                if gs is not None:
                    state[6] = gs.msg.width
            
            # 4. 右臂夹爪 (4. Right arm gripper)
            if self.right_gripper:
                gs = self.right_gripper.get_gripper_status()
                if gs is not None:
                    state[13] = gs.msg.width
                    
        except Exception as e:
            print(f"状态读取异常(status reading error)：{e}")
        return state
        
    def move(self, position_state):
        """
        移动到特定位置 
        (Move to specific position)
        """
        
        left_arm_position = position_state[0:6].tolist()
        right_arm_position = position_state[7:13].tolist()
        left_gripper_position = max(0.0, min(position_state[6], 0.1)) # width 0.0-0.1
        right_gripper_position = max(0.0, min(position_state[13], 0.1))
        try:
            # 左臂 left arm
            self.left_arm.move_js(left_arm_position)
            self.left_gripper.move_gripper(width=left_gripper_position, force=1.0)
            # 右臂 right arm
            self.right_arm.move_js(right_arm_position)
            self.right_gripper.move_gripper(width=right_gripper_position, force=1.0)
            
            time.sleep(0.02)
        except Exception as e:
            print(f"移动失败(move failed)：{e}")
    
    def move_initial(self, position_state):
        """
        移动到初始位置
        (Move to initial position)
        """
        n=50
        left_arm_position = position_state[0:6].tolist()
        right_arm_position = position_state[7:13].tolist()
        left_gripper_position = max(0.0, min(position_state[6], 0.1)) # width 0.0-0.1
        right_gripper_position = max(0.0, min(position_state[13], 0.1))
        left_traj = np.linspace(self.get_status_and_state()[0:6], left_arm_position, n)
        right_traj = np.linspace(self.get_status_and_state()[7:13], right_arm_position, n)
        try:
           for i in range(n):
                print(left_traj[i],right_traj[i])
                # 左臂 left arm
                self.left_arm.move_js(left_traj[i].tolist())
                self.left_gripper.move_gripper(width=left_gripper_position, force=1.0)
                # 右臂 right arm
                self.right_arm.move_js(right_traj[i].tolist())
                self.right_gripper.move_gripper(width=right_gripper_position, force=1.0)
                time.sleep(0.02)
        except Exception as e:
            print(f"移动失败(move failed)：{e}")

    def run(self, ros_operator, args):
        global success_count, total_count, stop_signal
        
        if not self.connect_arms(): return False
        print(self.left_arm.get_firmware())
        # 连接服务器 (Connect to server)
        self.client = websocket_client_policy.WebsocketClientPolicy("localhost", 8000)
        if not self.client: return False
        
        while True:
            # 1. 重置本轮保存变量 (Reset save variables for this round)
            save_states=[]
            save_actions = []
            save_images = []

            stop_signal.clear()  # 清除停止信号，允许 producer 工作(Clear stop signal to allow producer to work)
            _action_stop_event.clear()
            # 2. 回到初始位置 (Return to initial position)
            rate = rospy.Rate(100)
            rate.sleep()
            initial_position = np.concatenate((self.LEFT_INIT_POSITION, self.RIGHT_INIT_POSITION))
            self.move_initial(initial_position)

            # 3. 清空缓冲并启动producer线程 (Clear buffer and start producer thread)
            clear_action_buffer()
            global _action_prod_thread, _control_t
            _action_prod_thread = threading.Thread(target=self._action_producer_loop, args=(ros_operator,), daemon=True)
            _action_prod_thread.start()
            
            # 4. 等待开始 (Wait to start)
            input("按回车键开始模型推理(Press Enter to start model inference)...")

            # 5. 设置终端为非阻塞模式 (用于检测空格) (Set terminal to non-blocking mode (for detecting space))
            fd = sys.stdin.fileno()
            old_settings = termios.tcgetattr(fd)
            tty.setcbreak(fd)      
            
            t = 0
            last_action = initial_position
            st=time.time()
            try:
                while not rospy.is_shutdown() and not stop_signal.is_set():
                    with _action_lock:
                        _control_t = t
                    # Temporal Ensembling提取当前t动作 (Temporal Ensembling extracts current t action)
                    action = temporal_ensembled_action(t)
                    if action is None or len(action) == 0:
                        print(f"[WAITING] Waiting for action...")
                        continue
                    rate.sleep()
                    
                    # 6. 安全检查 (Safety check)
                    prev_curr_l1 = np.mean(np.abs(np.array(action) - last_action))
                    if prev_curr_l1 > self.ACTION_SAFETY_THRESHOLD:
                        print(f"\033[31m 安全检查失败：动作变化过大(Safety check failed: Action change too much) {prev_curr_l1:.3f}\033[0m")
                        break
                    
                    current_state = self.get_status_and_state()
                    # 检查动作与当前状态的差异 (防止跳变) (Check the difference between action and current state (prevent jumps))
                    prev_state_l1 = np.mean(np.abs(current_state - last_action))
                    if prev_state_l1 > self.STATE_SAFETY_THRESHOLD:
                        print(f"\033[31m 安全检查失败：状态差异过大(Safety check failed: State difference too large) {prev_state_l1:.3f}\033[0m")
                        break
                    
                    last_action = np.array(action)
                    
                    # 7. 记录数据 (Record data)
                    img_tuple = [ros_operator.img_front, ros_operator.img_left, ros_operator.img_right]
                    if img_tuple:
                        save_images.append(img_tuple) # 取前三个图片(take the first three images)
                        save_actions.append(action.copy())
                        save_states.append(current_state.copy())
                        print("保存长度：",len(save_images), len(save_actions), len(save_states))
                    # 8. 执行动作 (Execute action)
                    print(_control_t,t)
                    print("待执行(Pending execution) action_chunk:",action )
                    try:
                        self.move(action)
                        rate.sleep()
                    except Exception as e:
                        print(f"执行异常(Execute error)：{e}")
                    
                    t += 1
                    rate.sleep()

                    # 9. 检测空格键停止 (Detect space key to stop)
                    r, w, x = select.select([sys.stdin], [], [], 0)
                    if r:
                        key = sys.stdin.read(1)
                        if key == ' ':
                            print("\n结束单次测试()(End of this episode)\n")
                            ed=time.time()
                            cost_time=ed-st
                            print(f"本次耗时(cost time): {cost_time:.2f}s, {t}steps")
                            print(t/cost_time, " steps/s")
                            break
        
            except Exception as e:
                print(f"\nerror：{e}")
            finally:
                # 恢复终端设置(reset terminal settings)
                termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
                termios.tcflush(fd, termios.TCIFLUSH)

            # 10. 停止 producer 线程 (10. Stop producer thread)
            stop_signal.set() 
            _action_stop_event.set()
            if _action_prod_thread is not None and _action_prod_thread.is_alive():
                _action_prod_thread.join(timeout=2.0)
            
            # 11. 询问成功与否 (11. Ask if successful)
            user_input = input("Success? Enter 'y' or 'n': ")
            success = True if user_input.lower() == 'y' else False
            
            if success:
                success_count += 1
            total_count += 1
            
            # 12. 询问是否保存 (12. Ask if to save)
            user_input = input("Save this episode? Enter 'y'(success/fail), or 'n'(no): ")
            save = False
            folder_type = ""
            
            if user_input.lower() == 'y':
                save = True
                if success==True:
                    folder_type = "success"
                else:
                    folder_type = "fail"
            elif user_input.lower() == 'n':
                save = False
            
            if save:
                save_data(save_images, save_actions, save_states, success, folder_type, args)
            
            # 13. 统计成功率 (13. Calculate success rate)
            try:
                success_rate = success_count / total_count * 100
            except:
                success_rate = 0.0
            
            print("\n" + "="*50)
            print("当前测试次数(present rollout episodes):", total_count)
            print("此次是否成功(success?):", "是" if success else "否")
            print("当前成功率(success rate):", f"{success_rate:.2f}%")
            print("当前成功次数(present successful episodes):", success_count, "总测试次数(total episodes):", total_count)
            print("="*50 + "\n")
            
            # 14. 询问是否继续 (14. Ask if to continue)
            continue_input = input("继续下一次测试？(Continue to next episode?) Enter 'y' or 'n': ")
            if continue_input.lower() != 'y':
                break
        
        self._cleanup()

    def _action_producer_loop(self, ros_operator):
        """
        后台线程：采帧→请求server→写入历史缓存 
        (Background thread: capture frames → request server → write to history cache)
        """
        global _control_t, _action_chunks
        rate = rospy.Rate(100)  
        print_flag_local = True
        while not _action_stop_event.is_set() and not rospy.is_shutdown() and not stop_signal.is_set():
            with _action_lock:
                cursor = _control_t
                already_inferred = cursor in _action_chunks

            if already_inferred:
                rate.sleep()
                continue
            result = ros_operator.get_frame()
            if not isinstance(result, tuple) or len(result) < 6:
                if print_flag_local:
                    print("async syn fail")
                    print_flag_local = False
                rate.sleep()
                continue
            print_flag_local = True
            e=time.time()
            
            
            (img_h,img_l, img_r, img_h_depth, img_l_depth, img_r_depth) = result
            # resize before send to server (发送到服务器前调整大小)
            img_h_processed = image_tools.convert_to_uint8(
                image_tools.resize_with_pad(img_h,224,224).transpose(2,0,1)
            )
            img_l_processed = image_tools.convert_to_uint8(
                image_tools.resize_with_pad(img_l,224,224).transpose(2,0,1)
            )
            img_r_processed = image_tools.convert_to_uint8(
                image_tools.resize_with_pad(img_r,224,224).transpose(2,0,1)
            )
            
            # 构建obs (Build obs)
            obs={
                "images": {"cam_high": img_h_processed, 
                            "cam_left_wrist": img_l_processed, 
                            "cam_right_wrist": img_r_processed},
                "state": self.get_status_and_state(),
                "prompt": self.instruction
            }

            # === 调用OpenPI客户端推理 (Call OpenPI client inference) ===
            '''
            这里调用了openpi官方的本地服务器推理接口，如果不用服务器，把这里替换成本地推理代码即可。
            模型接收obs作为输入，输出14维动作序列。
            (Here we call the local server inference interface of OpenPI official. If you don't use the server, just replace it with local inference code here.
            The model receives obs as input and outputs a 14-dimensional action sequence.)
            '''
            try:
                action_chunk = self.client.infer(obs)["actions"]
                action_chunk = action_chunk[:self.ACTION_CHUNK_SIZE]
                with _action_lock:
                    current_t = _control_t
                _enqueue_chunk_to_expected_queues(action_chunk, cursor)
            except Exception as e:
                print(f"Inference error: {e}")
            rate.sleep()
            

    def _cleanup(self):
        """程序退出时的清理工作（统一管理） (Cleanup work when the program exits (unified management))"""
        print("\n正在退出(Exiting)...")
        _action_stop_event.set()
        
        if _action_prod_thread is not None and _action_prod_thread.is_alive():
            _action_prod_thread.join(timeout=2.0)
            print("producer线程已停止(producer thread stopped).")
        
        print("Inference stopped.")


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_front_topic', action='store', type=str, default='/camera_h/color/image_raw', required=False)
    parser.add_argument('--img_left_topic', action='store', type=str, default='/camera_l/color/image_raw', required=False)
    parser.add_argument('--img_right_topic', action='store', type=str, default='/camera_r/color/image_raw', required=False)

    parser.add_argument('--img_front_depth_topic', action='store', type=str, default='/camera_h/depth/image_raw', required=False)
    parser.add_argument('--img_left_depth_topic', action='store', type=str, default='/camera_l/depth/image_raw', required=False)
    parser.add_argument('--img_right_depth_topic', action='store', type=str, default='/camera_r/depth/image_raw', required=False)
    parser.add_argument('--use_depth_image', action='store', type=bool, default=False, required=False)

    parser.add_argument('--output_dir', action='store', type=str, 
                        default='<your_rollout_data_path>', required=False)

    args, unknown = parser.parse_known_args()

    return args


def main():
    # 注册 SIGINT 信号处理 (Ctrl+C) (Register SIGINT signal handler (Ctrl+C))
    def signal_handler(sig, frame):
        print("\n收到中断信号，正在清理(Receive interrupt signal, cleaning up)...")
        global stop_signal, _action_stop_event
        stop_signal.set()
        _action_stop_event.set()
        sys.exit(0)
        
    signal.signal(signal.SIGINT, signal_handler)
    args = get_arguments()
    ros_operator = RosOperator(args)
    try:
        controller = InferController(host="localhost", port=8000)
        controller.run(ros_operator,args)
    finally:
        pass

if __name__ == "__main__":
    main()
