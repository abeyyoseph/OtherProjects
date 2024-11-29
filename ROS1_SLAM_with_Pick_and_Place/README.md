This project focuses on several components of ROS1 and Gazebo. A world is first designed with walls, cube shaped obstacles, and tables.
A wheeled robot is then created that is capable of motion using the teleop_twist_keyboard node, or with SLAM by detecting and avoiding
obstacles using camera/LiDAR data. Additionally, two panda robot arms are created. The first arm is capable of picking up the cube from 
table. Problems with detaching the cube from the end effector could not be resolved.

To start project:
1. roslaunch panda_multiple_arms bringup_moveit.launch
2. rosrun moveit_tutorials right_arm_pick
3. rosservice call /link_attacher_node/attach "model_name_1: 'panda_multiple_arms'
link_name_1: 'right_arm_leftfinger'
model_name_2: 'cube_table'
link_name_2: 'my_cube'"
ok: True


For first time building the project:
catkin_make -DCMAKE_BUILD_TYPE=Release -DFranka_DIR:PATH=/path/to/libfranka/build
