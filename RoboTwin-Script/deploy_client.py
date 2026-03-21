#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import time
import signal
import numpy as np
import cv2
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
#==============================================================
class RosOperator:
    def __init__(self, args):
        self.img_front_deque = None
        self.img = None
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

    def img_right_callback(self, msg):
        if len(self.img_right_deque) >= 2000:
            self.img_right_deque.popleft()
        self.img_right_deque.append(msg)

    def img_front_callback(self, msg):
        if len(self.img_front_deque) >= 2000:
            self.img_front_deque.popleft()
        self.img_front_deque.append(msg)

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


class InferController:
    def __init__(self, host="localhost", port=8000):
        self.client = None
        self.left_channel = "can_left"
        self.right_channel = "can_right"
        self.bitrate = 1000000
        
        # === 机械臂参数 ===
        self.speed_pct = 50           # 速度百分比
        # self.max_linear_vel = 1.2     # m/s
        # self.max_angular_vel = 0.35   # rad/s
        # self.max_linear_acc = 0.8     # m/s2
        # self.max_angular_acc = 0.5    # rad/s2

        self.max_linear_vel = 1     # m/s
        self.max_angular_vel = 0.1   # rad/s
        self.max_linear_acc = 0.4     # m/s2
        self.max_angular_acc = 0.05    # rad/s2

        # === 夹爪对象 ===
        self.left_gripper = None
        self.right_gripper = None
        
        # === 指令 ===
        self.instruction = "fold the cloth"
        
        # === 执行参数 ===
        self.ACTION_CHUNK_SIZE = 50
        
        # === 初始位置 ===
        self.LEFT_INIT_POSITION = [-0.017, 1.293, -1.166, -0.323, 1.034, 0.486, 0.058]
        self.RIGHT_INIT_POSITION = [0.019, 1.565, -1.226, 0.4, 1.029, -0.258, 0.071]

    def connect_arms(self):
        """连接双臂并初始化夹爪"""
        print("连接双臂...")
        try:
            # --- 左臂 ---
            cfg_l = create_agx_arm_config(
                robot="piper", comm="can", channel=self.left_channel, bitrate=self.bitrate
            )
            self.left_arm = AgxArmFactory.create_arm(cfg_l)
            # 初始化夹爪
            self.left_gripper = self.left_arm.init_effector(
                self.left_arm.OPTIONS.EFFECTOR.AGX_GRIPPER
            )
            self.left_arm.connect()
            
            # --- 右臂 ---
            cfg_r = create_agx_arm_config(
                robot="piper", comm="can", channel=self.right_channel, bitrate=self.bitrate
            )
            self.right_arm = AgxArmFactory.create_arm(cfg_r)
            # 初始化夹爪
            self.right_gripper = self.right_arm.init_effector(
                self.right_arm.OPTIONS.EFFECTOR.AGX_GRIPPER
            )
            self.right_arm.connect()
            
            time.sleep(0.5)
            if not (self.left_arm.is_ok() and self.right_arm.is_ok()):
                raise Exception("机械臂连接状态检查失败")
            
            # 设置速度和使能
            for name, arm in [("左臂", self.left_arm), ("右臂", self.right_arm)]:
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
                    raise Exception(f"{name} 使能超时")
                
            print("连接成功\n")
            return True
            
        except Exception as e:
            print(f"连接错误：{e}")
            return False
    
    def get_status_and_state(self):
        """获取当前状态：12 维关节 (6+6) + 2 维夹爪 (归一化) = 14 维"""
        state = np.zeros(14, dtype=np.float32)
        try:
            # 1. 左臂关节
            ja_l = self.left_arm.get_joint_angles()
            if ja_l is not None: 
                state[0:6] = ja_l.msg
            
            # 2. 右臂关节
            ja_r = self.right_arm.get_joint_angles()
            if ja_r is not None: 
                state[7:13] = ja_r.msg
            
            # 3. 左臂夹爪
            if self.left_gripper:
                gs = self.left_gripper.get_gripper_status()
                if gs is not None:
                    state[6] = gs.msg.width
            
            # 4. 右臂夹爪
            if self.right_gripper:
                gs = self.right_gripper.get_gripper_status()
                if gs is not None:
                    state[13] = gs.msg.width
                    
        except Exception as e:
            print(f"状态读取异常：{e}")
        return state
    
    def move(self, position_state):
        """移动到特定位置"""
        
        left_arm_position = position_state[0:6].tolist()
        right_arm_position = position_state[7:13].tolist()
        left_gripper_position = max(0.0, min(position_state[6], 0.1)) # width 0.0-0.1
        right_gripper_position = max(0.0, min(position_state[13], 0.1))
        try:
            # 左臂
            self.left_arm.move_js(left_arm_position)
            self.left_gripper.move_gripper(width=left_gripper_position, force=1.0)
            # 右臂
            self.right_arm.move_js(right_arm_position)
            self.right_gripper.move_gripper(width=right_gripper_position, force=1.0)
            
            time.sleep(0.02)
        except Exception as e:
            print(f"移动失败：{e}")

    def move_initial(self, position_state):
        """移动到初始位置"""
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
                # 左臂
                self.left_arm.move_js(left_traj[i].tolist())
                self.left_gripper.move_gripper(width=left_gripper_position, force=1.0)
                # 右臂
                self.right_arm.move_js(right_traj[i].tolist())
                self.right_gripper.move_gripper(width=right_gripper_position, force=1.0)
                time.sleep(0.02)
        except Exception as e:
            print(f"移动失败：{e}")
    
    def run(self, ros_operator):
        if not self.connect_arms(): return False
        # 连接服务器
        self.client = websocket_client_policy.WebsocketClientPolicy("localhost", 8000)
        if not self.client: return False
        # 回到初始位置
        self.move_initial(np.concatenate((self.LEFT_INIT_POSITION, self.RIGHT_INIT_POSITION)))
        
        input("按回车键开始模型推理...")
        
        try:

            rate = rospy.Rate(50)
            print_flag_local = True
            while True:
                while not rospy.is_shutdown():
                    # 采帧（必要时等待）
                    result = ros_operator.get_frame()
                    if not result:
                        if print_flag_local:
                            print("async syn fail")
                            print_flag_local = False
                        rate.sleep()
                        continue
                    break
                print_flag_local = True
                (img_h,img_l, img_r, img_h_depth, img_l_depth, img_r_depth) = result
                
                # resize before send to server
                img_h_processed = image_tools.convert_to_uint8(
                    image_tools.resize_with_pad(img_h,224,224).transpose(2,0,1)
                )
                img_l_processed = image_tools.convert_to_uint8(
                    image_tools.resize_with_pad(img_l,224,224).transpose(2,0,1)
                )
                img_r_processed = image_tools.convert_to_uint8(
                    image_tools.resize_with_pad(img_r,224,224).transpose(2,0,1)
                )
                
                # img_h_processed = image_tools.convert_to_uint8(
                #     img_h.transpose(2,0,1)
                # )
                # img_l_processed = image_tools.convert_to_uint8(
                #     img_l.transpose(2,0,1)
                # )
                # img_r_processed = image_tools.convert_to_uint8(
                #     img_r.transpose(2,0,1)
                # )
                
                # 构建obs
                obs={
                    "images": {"cam_high": img_h_processed, 
                               "cam_left_wrist": img_l_processed, 
                               "cam_right_wrist": img_r_processed},
                    "state": self.get_status_and_state(),
                    "prompt": self.instruction
                }
                # obs = self.build_observation()
                # 推理
                print("开始推理...\n")
                action_chunk = self.client.infer(obs)["actions"]
                # 执行前self.ACTION_CHUNK_SIZE步
                action_chunk = action_chunk[:self.ACTION_CHUNK_SIZE]
                print("待执行action_chunk:",action_chunk )
                for i in range(len(action_chunk)):
                    action = action_chunk[i] # Shape: 14
                    try:
                        self.move(action)
                        time.sleep(0.05)
                    except Exception as e:
                        print(f"Step{i} 异常：{e}")
                
        except Exception as e:
            print(f"\n错误：{e}")
def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_front_topic', action='store', type=str, default='/camera_h/color/image_raw', required=False)
    parser.add_argument('--img_left_topic', action='store', type=str, default='/camera_l/color/image_raw', required=False)
    parser.add_argument('--img_right_topic', action='store', type=str, default='/camera_r/color/image_raw', required=False)

    parser.add_argument('--img_front_depth_topic', action='store', type=str, default='/camera_h/depth/image_raw', required=False)
    parser.add_argument('--img_left_depth_topic', action='store', type=str, default='/camera_l/depth/image_raw', required=False)
    parser.add_argument('--img_right_depth_topic', action='store', type=str, default='/camera_r/depth/image_raw', required=False)
    parser.add_argument('--use_depth_image', action='store', type=bool, default=False, required=False)

    # args = parser.parse_args()
    args, unknown = parser.parse_known_args()

    return args

def main():
    signal.signal(signal.SIGINT, lambda s,f: sys.exit(0))
    args = get_arguments()
    ros_operator = RosOperator(args)
    try:
        InferController().run(ros_operator)
    finally:
        pass

if __name__ == "__main__":
    main()