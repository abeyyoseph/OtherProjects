This project focuses on analyzing the performance of three reinforcement learning algorithms, Q-learning/DQN/DDQN, in the Gazebo physics simulator
for ROS applications. A wheeled robot is initialized in an empty Gazebo room with four walls and integrated with the python Gym/openai_ros libraries
to control its movements in the environment. An incentive structure was designed to focus the robot on moving towards a randomly assigned goal 
location. Once the robot had been trained in the empty environment, it was moved into an environment with an obstacle. The incentive structure was 
updated to penalize the robot if it got too close to the obstacle. Performance was quantitatively measured based on the number of times the robot reached
the goal position and qualitatively measured based on how well the robot explored its environment.

To start project:
1. roslaunch gym_setup main.launch
2. roslaunch atom_training start_training_wall.launch

For first time building the project:
catkin_make -DCMAKE_BUILD_TYPE=Release -DFranka_DIR:PATH=/path/to/libfranka/build
