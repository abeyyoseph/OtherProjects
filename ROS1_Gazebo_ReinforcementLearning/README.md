# Reinforcement Learning for Robot Navigation in Gazebo

This project analyzes the performance of three reinforcement learning algorithms—**Q-learning, DQN, and DDQN**—in the **Gazebo** physics simulator for **ROS** applications. 

A **wheeled robot** is initialized in an empty Gazebo room with four walls and is integrated with the **Gym/OpenAI_ROS** libraries to control its movements. The goal is to train the robot to navigate towards a randomly assigned goal location while avoiding obstacles.

---

## IEEE Paper
The full writeup for this project entitled "Comparative Analysis of Reinforcement Learning Techniques in Simulated Robotics Environments" can be found here: https://ieeexplore.ieee.org/document/10739810

## Project Overview

1. **Environment Setup**
   - The robot starts in an empty Gazebo environment with four walls.
   - Integrated with **Gym/OpenAI_ROS** for reinforcement learning.

2. **Incentive Structure**
   - The robot receives positive rewards for moving towards a **random goal**.
   - After training in the empty room, an **obstacle** is introduced.
   - The incentive structure is updated to penalize the robot for getting too close to the obstacle.

3. **Performance Evaluation**
   - **Quantitative**: The number of times the robot successfully reaches the goal.
   - **Qualitative**: Observing how well the robot explores its environment.

---

## Getting Started

### Running the Project

To launch the simulation:

```bash
roslaunch gym_setup main.launch
roslaunch atom_training start_training_wall.launch

For first time building the project:
catkin_make -DCMAKE_BUILD_TYPE=Release -DFranka_DIR:PATH=/path/to/libfranka/build
