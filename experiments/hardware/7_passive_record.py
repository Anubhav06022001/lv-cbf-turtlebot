#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
import cv2
import time
import csv
import numpy as np

class PassiveRecorder(Node):
    def __init__(self):
        super().__init__('passive_recorder')
        # Listen to whatever your control script is publishing
        self.sub_cmd = self.create_subscription(Twist, '/HWTB3_3/cmd_vel', self.cmd_callback, 10)
        self.sub_odom = self.create_subscription(Odometry, '/HWTB3_3/odom', self.odom_callback, 10)
        
        self.current_v, self.current_w = 0.0, 0.0
        self.pos_x, self.pos_y, self.yaw = 0.0, 0.0, 0.0

        # Open RealSense RGB
        self.cap = cv2.VideoCapture(4)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.out_video = cv2.VideoWriter('lvcbf_camera_run.mp4', fourcc, 10.0, (640, 480))
        
        self.csv_file = open('lvcbf_camera_data.csv', 'w', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow(['timestamp', 'v_forward', 'w_yaw', 'pos_x', 'pos_y', 'yaw'])

        print("Camera is ON! Waiting for your script to move the robot...")
        self.timer = self.create_timer(0.1, self.record_frame)

    def cmd_callback(self, msg):
        self.current_v = msg.linear.x
        self.current_w = msg.angular.z
        
    def odom_callback(self, msg):
        self.pos_x = msg.pose.pose.position.x
        self.pos_y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        self.yaw = np.arctan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))

    def record_frame(self):
        ret, frame = self.cap.read()
        if ret:
            self.out_video.write(frame)
            self.csv_writer.writerow([time.time(), self.current_v, self.current_w, self.pos_x, self.pos_y, self.yaw])

def main(args=None):
    rclpy.init(args=args)
    node = PassiveRecorder()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.cap.release()
        node.out_video.release()
        node.csv_file.close()
        node.destroy_node()
        rclpy.shutdown()
        print("Saved lvcbf_camera_run.mp4!")

if __name__ == '__main__':
    main()