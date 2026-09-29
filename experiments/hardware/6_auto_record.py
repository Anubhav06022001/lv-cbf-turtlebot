#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
import numpy as np
import cv2
import time
import csv

class AutoDemoRecorder(Node):
    def __init__(self):
        super().__init__('auto_demo_recorder')
        
        # --- Navigation Setup ---
        self.publisher_ = self.create_publisher(Twist, '/HWTB3_3/cmd_vel', 10)
        self.subscription = self.create_subscription(Odometry, '/HWTB3_3/odom', self.odom_callback, 10)
        
        self.waypoints = [
            np.array([1.5, 0.0]), 
            np.array([2.0, 0.35]),  
            np.array([2.5, 0.0]),  
            np.array([4.0, 0.0])   
        ]
        self.current_wp = 0
        self.current_x = 0.0
        self.current_y = 0.0
        self.current_yaw = 0.0
        self.odom_received = False
        
        # --- Camera & Logging Setup ---
        self.cap = cv2.VideoCapture(4)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        if not self.cap.isOpened():
            print("ERROR: Could not open camera at index 4.")
            return

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.out_video = cv2.VideoWriter('autonomous_demo.mp4', fourcc, 10.0, (640, 480))
        
        self.csv_file = open('autonomous_demo_data.csv', 'w', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow(['timestamp', 'v_forward', 'w_yaw', 'pos_x', 'pos_y', 'yaw'])

        print("Autonomous Recording started! Waiting for odometry...")
        
        # Run control and recording math at 10Hz
        self.timer = self.create_timer(0.1, self.control_and_record_loop)

    def odom_callback(self, msg):
        self.current_x = msg.pose.pose.position.x
        self.current_y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        self.current_yaw = np.arctan2(siny_cosp, cosy_cosp)
        self.odom_received = True

    def control_and_record_loop(self):
        if not self.odom_received:
            return
            
        v = 0.0
        w = 0.0
            
        if self.current_wp >= len(self.waypoints):
            self.send_vel(0.0, 0.0)
            print("Goal Reached! Press Ctrl+C to stop recording and save.")
        else:
            target = self.waypoints[self.current_wp]
            pos_2d = np.array([self.current_x, self.current_y])
            dist = np.linalg.norm(target - pos_2d)
            
            if dist < 0.15:
                print(f"*** Reached Waypoint {self.current_wp} ***")
                self.current_wp += 1
                self.send_vel(0.0, 0.0)
            else:
                target_yaw = np.arctan2(target[1] - self.current_y, target[0] - self.current_x)
                yaw_err = np.arctan2(np.sin(target_yaw - self.current_yaw), np.cos(target_yaw - self.current_yaw))
                
                # Steering logic
                if abs(yaw_err) > 0.5:
                    v = 0.0
                    w = 2.0 * np.sign(yaw_err)
                else:
                    v = 0.15 
                    w = 2.5 * yaw_err
                    
                self.send_vel(v, w)
                print(f"WP: {self.current_wp} | x: {self.current_x:.2f}, y: {self.current_y:.2f} | v: {v:.2f}, w: {w:.2f}")

        # --- Record Frame & State ---
        ret, frame = self.cap.read()
        if ret:
            self.out_video.write(frame)
            self.csv_writer.writerow([time.time(), v, w, self.current_x, self.current_y, self.current_yaw])

    def send_vel(self, v, w):
        msg = Twist()
        msg.linear.x = float(v)
        msg.angular.z = float(w)
        self.publisher_.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = AutoDemoRecorder()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    finally:
        node.send_vel(0.0, 0.0) # Failsafe stop
        if hasattr(node, 'cap'):
            node.cap.release()
            node.out_video.release()
            node.csv_file.close()
        node.destroy_node()
        rclpy.shutdown()
        print("Saved autonomous_demo.mp4 and autonomous_demo_data.csv")

if __name__ == '__main__':
    main()