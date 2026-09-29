# #!/usr/bin/env python3
# import rclpy
# from rclpy.node import Node
# from geometry_msgs.msg import Twist
# import csv
# import time
# from pathlib import Path

# class CSVReplayNode(Node):
#     def __init__(self):
#         super().__init__('csv_replay_node')
#         self.publisher_ = self.create_publisher(Twist, '/HWTB3_3/cmd_vel', 10)
#         project_root = Path(__file__).resolve().parent.parent.parent
#         self.csv_path = project_root / 'results' / 'csv' / 'baseline_waypoint_controls.csv'
        
#     def run_replay(self):
#         try:
#             with open(str(self.csv_path), 'r') as file:
#                 reader = csv.DictReader(file)   
#                 for index, row in enumerate(reader):
#                     # Downsample: Only process every 50th row to convert 500Hz to 10Hz
#                     if index % 50 != 0:
#                         continue
                        
#                     msg = Twist()
#                     msg.linear.x = (float(row['v']) - 0.02)
#                     msg.angular.z = float(row['w'])
                    
#                     # Extract the simulation time directly from the CSV
#                     sim_time = float(row['time'])
                    
#                     self.publisher_.publish(msg)
#                     print(f"Time: {sim_time:.2f}s | Sending -> v: {msg.linear.x:.2f}, w: {msg.angular.z:.2f}")
                    
#                     # Sleep for 0.1 seconds (10Hz)
#                     time.sleep(0.1) 
                    
#         except FileNotFoundError:
#             print(f"Error: Could not find CSV file at {self.csv_path}")
#         finally:
#             stop_msg = Twist()
#             self.publisher_.publish(stop_msg)
#             print("CSV finished. Stopping robot.")

# def main(args=None):
#     rclpy.init(args=args)
#     node = CSVReplayNode()
#     node.run_replay()
#     node.destroy_node()
#     rclpy.shutdown()

# if __name__ == '__main__':
#     main()

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
        self.publisher_ = self.create_publisher(Twist, '/HWTB3_3/cmd_vel', 10)
        project_root = Path(__file__).resolve().parent.parent.parent
        self.csv_path = project_root / 'results' / 'csv' / 'baseline_waypoint_controls.csv'
        
    def run_replay(self):
        try:
            with open(str(self.csv_path), 'r') as file:
                reader = csv.DictReader(file)   
                for row in reader:
                    msg = Twist()
                    # Directly assign v and w from your new, clean dataset!
                    msg.linear.x = float(row['v'])
                    msg.angular.z = float(row['w'])
                    
                    self.publisher_.publish(msg)
                    print(f"Time: {row['time']}s | Sending -> v: {msg.linear.x:.2f}, w: {msg.angular.z:.2f}")
                    
                    # Exact 10Hz timing matching your new CSV
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