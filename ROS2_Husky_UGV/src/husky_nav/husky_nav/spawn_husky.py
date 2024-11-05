import rclpy
from rclpy.node import Node
from gazebo_msgs.srv import SpawnEntity
import os

class SpawnHusky(Node):
    def __init__(self):
        super().__init__('spawn_husky')
        client = self.create_client(SpawnEntity, '/spawn_entity')
        client.wait_for_service()

        request = SpawnEntity.Request()
        request.name = 'a200'
        print("Attempting to open the a200 urdf file.")
        request.xml = open('src/husky_nav/husky_nav/husky/scout_ros2/scout_description/urdf/scout_v2.xacro', 'r').read()
        client.call_async(request)
        print("Spawning the a200.")


def main(args=None):
    rclpy.init(args=args)
    node = SpawnHusky()
    rclpy.spin(node)
    rclpy.shutdown()
