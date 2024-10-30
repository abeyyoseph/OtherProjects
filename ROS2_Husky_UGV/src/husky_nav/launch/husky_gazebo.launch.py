# husky_gazebo.launch.py

from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    # Get the directory where Husky's Gazebo launch files are located
    husky_gazebo_dir = get_package_share_directory('husky_gazebo')

    # Define the path to the Husky Gazebo launch file
    gazebo_launch_path = os.path.join(husky_gazebo_dir, 'launch', 'gazebo.launch.py')

    # Include the Husky Gazebo launch file
    gazebo_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(gazebo_launch_path)
    )

    return LaunchDescription([
        gazebo_launch,
    ])
