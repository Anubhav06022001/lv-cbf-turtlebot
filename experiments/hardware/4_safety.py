#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry

import sys
import numpy as np
import torch
from pathlib import Path

# --- Setup Paths & Imports ---
THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent.parent 
sys.path.insert(0, str(PROJECT_ROOT))

from src.vocbf.vocbf_network import BarrierNet
from src.control.cbf_qp_multi import cbf_qp_multi 

# CPU for Raspberry Pi compatibility
DEVICE = torch.device("cpu")

MEAN_PATH = PROJECT_ROOT / "models" / "tb3_x_mean.npy"
STD_PATH = PROJECT_ROOT / "models" / "tb3_x_std.npy"
MODEL_PATH = PROJECT_ROOT / "models" / "tb3_vocbf.pt"

GOAL_POS = np.array([4.0, 0.0])
ALPHA = 4.0
SAFETY_OFFSET = 0.10

class LiveVOCBFController(Node):
    def __init__(self):
        super().__init__('live_vocbf_controller')
        
        # ROS 2 Pub/Sub
        self.publisher_ = self.create_publisher(Twist, '/HWTB3_3/cmd_vel', 10)
        self.subscription = self.create_subscription(Odometry, '/HWTB3_3/odom', self.odom_callback, 10)
        
        # State Variables
        self.pos_2d = np.array([0.0, 0.0])
        self.yaw = 0.0
        self.v_forward = 0.0
        self.w_yaw = 0.0
        self.odom_received = False
        
        # Load CBF Model & Normalization
        self.x_mean = np.load(MEAN_PATH)
        self.x_std = np.load(STD_PATH)
        
        self.B_net = BarrierNet(state_dim=5).to(DEVICE)
        checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
        self.B_net.load_state_dict(checkpoint.get('model_state_dict', checkpoint))
        self.B_net.eval()
        
        # Control Loop (10Hz)
        self.timer = self.create_timer(0.1, self.control_loop)
        
    def odom_callback(self, msg):
        self.pos_2d[0] = msg.pose.pose.position.x
        self.pos_2d[1] = msg.pose.pose.position.y
        
        q = msg.pose.pose.orientation
        self.yaw = np.arctan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
        
        self.v_forward = msg.twist.twist.linear.x
        self.w_yaw = msg.twist.twist.angular.z
        self.odom_received = True

    def control_loop(self):
        if not self.odom_received:
            return
            
        state_5d = np.array([self.pos_2d[0], self.pos_2d[1], self.yaw, self.v_forward, self.w_yaw])
        
        # Check if goal reached
        if np.linalg.norm(GOAL_POS - self.pos_2d) < 0.15:
            self.send_vel(0.0, 0.0)
            print("Goal Reached! Stopping.")
            return

        # 1. Base Controller
        direction = GOAL_POS - self.pos_2d
        target_yaw = np.arctan2(direction[1], direction[0])
        yaw_err = np.arctan2(np.sin(target_yaw - self.yaw), np.cos(target_yaw - self.yaw))
        
        v_des = 0.15 * max(0.0, np.cos(yaw_err))
        w_des = 1.5 * yaw_err
        u_des = np.array([v_des, w_des, 0.0]) 
        
        # 2. CBF Neural Net Inference (State-Only)
        x_norm = (state_5d - self.x_mean) / self.x_std
        x_torch = torch.tensor(x_norm, dtype=torch.float32, requires_grad=True, device=DEVICE).unsqueeze(0)
        
        B_val = self.B_net(x_torch)
        grad_B_norm = torch.autograd.grad(B_val, x_torch)[0].cpu().numpy()[0]
        grad_B_pos = (grad_B_norm / self.x_std) 
        
        L_g_B_v = (grad_B_pos[0] * np.cos(self.yaw)) + (grad_B_pos[1] * np.sin(self.yaw))
        L_g_B_w = grad_B_pos[2]
        grad_B_ctrl = np.array([L_g_B_v, L_g_B_w, 0.0])
        
        B_raw = B_val.item()
        B_calibrated = B_raw - SAFETY_OFFSET
        
        # 3. Safe QP Solve
        A_list, b_list = [grad_B_ctrl], [-ALPHA * B_calibrated]
        u_safe = cbf_qp_multi(u_des, A_list, b_list)
        
        # 4. Hardware Safety Clamp (Max 0.15 m/s)
        v_safe = u_safe[0]
        w_safe = u_safe[1]
        
        self.send_vel(v_safe, w_safe)
        # print(f"B_raw: {B_raw:.4f} | u_des: [{u_des[0]:.2f}, {u_des[1]:.2f}] | u_safe: [{v_safe:.2f}, {w_safe:.2f}]")
        print(f"x: {self.pos_2d[0]:.2f}, y: {self.pos_2d[1]:.2f} | B_raw: {B_raw:.4f} | u_des: [{u_des[0]:.2f}, {u_des[1]:.2f}] | u_safe: [{v_safe:.2f}, {w_safe:.2f}]")
        


    def send_vel(self, v, w):
        msg = Twist()
        msg.linear.x = float(v)
        msg.angular.z = float(w)
        self.publisher_.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = LiveVOCBFController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # Failsafe wrap
        node.send_vel(0.0, 0.0)
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()