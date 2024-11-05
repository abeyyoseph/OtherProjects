from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
import os

def generate_launch_description():
    gazebo_env_path = os.getenv('GAZEBO_RESOURCE_PATH')
    gazebo_ros_paths = [path for path in gazebo_env_path.split(':')]
    
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([os.path.join(gazebo_ros_paths[1], 'launch', 'gazebo.launch.py')]),
        launch_arguments={'world': 'empty.world'}.items(),
    )

    spawn_husky = Node(
        package='husky_nav',
        executable='spawn_husky',
        output='screen'
    )

    return LaunchDescription([
        gazebo,
        spawn_husky,
    ])
