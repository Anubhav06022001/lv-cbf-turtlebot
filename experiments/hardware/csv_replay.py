#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
import csv
import time
from pathlib import Path

class CSVReplayNode(Node):
    def __init__(self):
        super().__init__('csv_replay_node')
        # Matches your lab mate's topic namespace
        self.publisher_ = self.create_publisher(Twist, '/HWTB3_3/cmd_vel', 10)
        # Use relative path from project root
        project_root = Path(__file__).resolve().parent.parent.parent
        self.csv_path = project_root / 'results' / 'csv' / 'baseline_waypoint_controls.csv'
        
    def run_replay(self):
        try:
            with open(str(self.csv_path), 'r') as file:
                reader = csv.DictReader(file)   
                for row in reader:
                    msg = Twist()
                    msg.linear.x = float(row['v'])
                    msg.angular.z = float(row['w'])
                    
                    self.publisher_.publish(msg)
                    print(f"Sending -> v: {msg.linear.x:.2f}, w: {msg.angular.z:.2f}")
                    
                    # Sleeps for 0.1 seconds (10Hz). Change this if you recorded at a different speed.
                    time.sleep(0.1) 
                    
        except FileNotFoundError:
            print(f"Error: Could not find CSV file at {self.csv_path}")
        finally:
            # Failsafe: stop motors when CSV ends
            stop_msg = Twist()
            self.publisher_.publish(stop_msg)
            print("CSV finished. Stopping robot.")

def main(args=None):
    rclpy.init(args=args)
    node = CSVReplayNode()
    node.run_replay()
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()