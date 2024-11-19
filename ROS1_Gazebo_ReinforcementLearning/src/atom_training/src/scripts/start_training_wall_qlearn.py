#!/usr/bin/env python3
import sys
sys.path.append('/home/abey/Desktop/Repos/Data_Analysis_Scripts/KSU_GradSchool/MTRE6800_ResearchProject/src/')
import openai_ros
import gym
import numpy
import time
import qlearn
from gym import wrappers
# ROS packages required
import rospy
import rospkg
import matplotlib.pyplot as plt
# import our training environment
from openai_ros.task_envs import atom_wall
import random

if __name__ == '__main__':

    print("Initializing the atom training node \n")

    rospy.init_node('atom_wall_qlearn', anonymous=True, log_level=rospy.WARN)

    # Create the Gym environment
    env = gym.make('AtomTrainingEnv-v0')
    print("Gym environment done")

    # Set the logging system
    rospack = rospkg.RosPack()
    pkg_path = rospack.get_path('atom_training')
    outdir = pkg_path + '/training_results'

    last_time_steps = numpy.ndarray(0)

    # Loads parameters from the ROS param server
    # Parameters are stored in a yaml file inside the config directory
    # They are loaded at runtime by the launch file
    Alpha = rospy.get_param("/atom/alpha")
    Epsilon = rospy.get_param("/atom/epsilon")
    Gamma = rospy.get_param("/atom/gamma")
    epsilon_discount = rospy.get_param("/atom/epsilon_discount")
    nepisodes = rospy.get_param("/atom/nepisodes")
    nsteps = rospy.get_param("/atom/nsteps")
    running_step = rospy.get_param("/atom/running_step")

    # Initialises the algorithm that we are going to use for learning
    qlearn = qlearn.QLearn(actions=range(env.action_space.n),
                           alpha=Alpha, gamma=Gamma, epsilon=Epsilon)

    qlearn.loadModelState()

    start_time = time.time()
    highest_reward = 0
    state_visit_count = numpy.zeros((10, 10))

    # Starts the main training loop 
    for x in range(nepisodes):

        cumulativeReward = 0
        done = False
        #Decrease the exploration rate at each episode resulting in more weight being applied to previous knowledge
        if qlearn.epsilon > 0.05:
            qlearn.epsilon *= epsilon_discount
        
        while True:
            goal_x = random.randint(-2, 3)
            goal_y = random.randint(0, 6)

            #Agent starting position at episode (-3, 3)
            if goal_y <= 2 or goal_y >= 4:
                break
        print("Generating random x,y goal position:" + str(goal_x) + " " + str(goal_y))

        # Initialize the environment and get first state of the robot
        env.resetCustom(goal_x, goal_y)
        observation = env.reset()

        state = tuple([observation[0], observation[1], observation[2], observation[3]])
        state_visit_count[(observation[4]+5), (observation[5]+2)] += 1

        print("STATE: " + str(state))
        rospy.logwarn("Current Epsilon:" + str(qlearn.epsilon))

        # for each episode, we train the robot for nsteps
        for i in range(nsteps):
            rospy.logwarn("Start Episode:" + str(x) + " step:" + str(i))
            # Pick an action based on the current state
            action = qlearn.chooseAction(state)

            # Execute the action in the environment and get feedback
            observation, reward, done, info = env.step(action)

            cumulativeReward += reward
            if highest_reward < cumulativeReward:
                highest_reward = cumulativeReward

            nextState = tuple([observation[0], observation[1], observation[2], observation[3]])
            state_visit_count[(observation[4]+5), (observation[5]+2)] += 1

            qlearn.learn(state, action, reward, nextState)

            if not (done):
                state = nextState
            else:
                last_time_steps = numpy.append(last_time_steps, [int(i + 1)])
                break
            rospy.logwarn("New state: " + str(state))
            rospy.logwarn("End Episode:" + str(x) + " step:" + str(i))

        episodeTrainingTime = int(time.time() - start_time)
        rospy.logwarn("Episode " + str(x) + " training time (sec): " + str(episodeTrainingTime))
        rospy.logwarn("Episode " + str(x) + " cumulative reward: " + str(cumulativeReward))

        if x % 25 == 0:
            plt.figure(figsize=(10, 8))
            plt.imshow(state_visit_count, cmap='viridis', interpolation='nearest')
            plt.colorbar()
            plt.title("State Visitation Heatmap Episode " + str(x))
            plt.savefig(f'/home/abey/Desktop/Repos/Data_Analysis_Scripts/KSU_GradSchool/MTRE6800_ResearchProject/data/heatmap_episode_' + str(x) + '.png')

    qlearn.save_state()

    env.close()

    
