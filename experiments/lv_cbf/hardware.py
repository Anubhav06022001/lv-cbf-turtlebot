import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
import time

class ReplayNode(Node):
    def __init__(self):
        super().__init__('replay_node')
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.timer = self.create_timer(0.1, self.run_replay)
        self.start_time = time.time()

    def run_replay(self):
        twist = Twist()
        if time.time() - self.start_time < 3.0:
            twist.linear.x = 0.1  # Move forward 0.1 m/s
            twist.angular.z = 0.0
            print("Driving forward...")
        else:
            twist.linear.x = 0.0  # Stop
            twist.angular.z = 0.0
            print("Stopping.")
            
        self.cmd_pub.publish(twist)

def main(args=None):
    rclpy.init(args=args)
    node = ReplayNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.cmd_pub.publish(Twist()) # Failsafe stop
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()