#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
import time

class StraightTest(Node):
    def __init__(self):
        super().__init__('straight_test')
        self.publisher_ = self.create_publisher(Twist, '/HWTB3_3/cmd_vel', 10)
        
    def test_motors(self):
        msg = Twist()
        msg.linear.x = -0.15  # Changed to 0.15 to respect hardware limits
        msg.angular.z = 0.0 
        
        print("Sending PURE FORWARD command (v=0.15, w=0.0) for 3 seconds...")
        print("Press Ctrl+C to stop early.")
        
        try:
            # 30 loops * 0.1s = 3 seconds
            for _ in range(300):
                self.publisher_.publish(msg)
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("\nTest interrupted by user! Stopping...")
        finally:
            # This block ALWAYS runs, even if you hit Ctrl+C
            stop_msg = Twist()
            self.publisher_.publish(stop_msg)
            print("Test finished. Motors stopped.")

def main(args=None):
    rclpy.init(args=args)
    node = StraightTest()
    node.test_motors()
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()