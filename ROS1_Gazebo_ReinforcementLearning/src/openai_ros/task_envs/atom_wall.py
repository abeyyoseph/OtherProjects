import rospy
import numpy
from gym import spaces
from geometry_msgs.msg import Point
from openai_ros.robot_envs import atom_env
from gym.envs.registration import register
from geometry_msgs.msg import PoseWithCovarianceStamped
from math import sin, cos
from gazebo_msgs.srv import SetModelState
from gazebo_msgs.msg import ModelState
from geometry_msgs.msg import Twist
import tf.transformations
import math 

#Max locations in the map
ENV_X_MIN = -10
ENV_X_MAX = 10
ENV_Y_MIN = -10
ENV_Y_MAX = 10
OBSTACLE_1_X = 2
OBSTACLE_1_Y = 4
OBSTACLE_2_X = -1
OBSTACLE_2_Y = 1

register(
        id='AtomTrainingEnv-v0',
        entry_point='openai_ros.task_envs.atom_wall:AtomWallEnv',
    )

class AtomWallEnv(atom_env.AtomEnv):
    def __init__(self):
        
        # Only variable needed to be set here
        number_actions = rospy.get_param('/atom/n_actions')
        self.action_space = spaces.Discrete(number_actions)

        # We set the reward range, which is not compulsory but here we do it.
        self.reward_range = (-numpy.inf, numpy.inf)

        # Actions and Observations
        self.linear_forward_speed = rospy.get_param('/atom/linear_forward_speed')
        self.linear_turn_speed = rospy.get_param('/atom/linear_turn_speed')
        self.angular_speed = rospy.get_param('/atom/angular_speed')
        self.init_linear_forward_speed = rospy.get_param('/atom/init_linear_forward_speed')
        self.init_linear_turn_speed = rospy.get_param('/atom/init_linear_turn_speed')

        self.new_ranges = rospy.get_param('/atom/new_ranges')
        self.min_range = rospy.get_param('/atom/min_range')
        self.max_laser_value = rospy.get_param('/atom/max_laser_value')
        self.min_laser_value = rospy.get_param('/atom/min_laser_value')

        self.grid_size = 1.0

        # Get Desired Point to Get
        self.desired_point = Point()
        self.desired_point_discretized = Point()
        self.desired_point_normalized = Point()

        # self.desired_point_discretized.x, self.desired_point_discretized.y = self.discretizePoint(self.desired_point.x, self.desired_point.y, 
        #                                                                                           self.grid_size)
        # self.desired_point_discretized.z = 0

        # We create two arrays based on the binary values that will be assigned
        # In the discretization method.
        laser_scan = self._check_laser_scan_ready()
        num_laser_readings = len(laser_scan.ranges) / self.new_ranges
        high = numpy.full((int(num_laser_readings)), self.max_laser_value)
        low = numpy.full((int(num_laser_readings)), self.min_laser_value)

        # We only use two integers
        self.observation_space = spaces.Box(low, high)

        rospy.logdebug("ACTION SPACES TYPE===>" + str(self.action_space))
        rospy.logdebug("OBSERVATION SPACES TYPE===>" + str(self.observation_space))

        # Rewards
        self.forwards_reward = rospy.get_param("/atom/forwards_reward")
        self.heading_reward = rospy.get_param("/atom/heading_reward")
        self.end_episode_points = rospy.get_param("/atom/end_episode_points")

        self.cumulative_steps = 0.0

        super(AtomWallEnv, self).__init__()


    def discretizePoint(self, x, y, grid_size):
        # discretized_x = round(x / grid_size)
        # discretized_y = round(y / grid_size)
        discretized_x = math.floor(x * 4) / 4
        discretized_y = math.floor(y * 4) / 4
        return [discretized_x, discretized_y]
    
    def normalize_feature(self, value, min_possible_value, max_possible_value):
        # First, normalize to [0, 1]
        normalized_value = (value - min_possible_value) / (max_possible_value - min_possible_value)
        # Then, scale to [-1, 1]
        scaled_value = (normalized_value * 2) - 1
        return round(scaled_value, 2)

    def normalize_state(self, x, y, rel_x, rel_y, env_x_min, env_x_max, env_y_min, env_y_max):
        """
        Normalize the agent's state features: absolute positions and relative positions to the goal.
        
        Assumes that relative positions could span the same range as the absolute positions.
        """
        normalized_x = self.normalize_feature(x, env_x_min, env_x_max)
        normalized_y = self.normalize_feature(y, env_y_min, env_y_max)
        normalized_rel_x = self.normalize_feature(rel_x, env_x_min - env_x_max, env_x_max - env_x_min)
        normalized_rel_y = self.normalize_feature(rel_y, env_y_min - env_y_max, env_y_max - env_y_min)

        return [normalized_x, normalized_y, normalized_rel_x, normalized_rel_y]

    def _set_init_pose(self):
        """Sets the Robot in its init pose
        """
        print("Publishing stop command")
        stop_command = Twist()  # All zero values by default
        pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        rate = rospy.Rate(10)  # 10hz
        for _ in range(10):
            pub.publish(stop_command)
            rate.sleep()

        try:
            print("Moving atom back to starting state")
            set_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
            state_msg = ModelState()
            state_msg.model_name = "wheeled_robot1"
            state_msg.pose.position.x = -3
            state_msg.pose.position.y = 3
            # Again, assuming theta is yaw, convert to quaternion if necessary
            # Example for theta=0; adjust as needed
            state_msg.pose.orientation.w = 1

            resp = set_state(state_msg)
            if not resp.success:
                rospy.logerr("Failed to set model state: " + resp.status_message)
            else:
                rospy.loginfo("Model state set successfully")
        except rospy.ServiceException as e:
            rospy.logerr("Service call failed: %s"%e)

        return True

    def _init_env_variables_custom(self, goal_x, goal_y):
        """
        Inits variables needed to be initialised each time we reset at the start
        of an episode.
        :return:
        """
        self.desired_point_discretized.x, self.desired_point_discretized.y = self.discretizePoint(goal_x, goal_y, 
                                                                                                  self.grid_size)
        self.desired_point_discretized.z = 0
        # self.desired_point_normalized.x = self.normalize_feature(goal_x, ENV_X_MIN, ENV_X_MAX)
        # self.desired_point_normalized.y = self.normalize_feature(goal_y, ENV_Y_MIN, ENV_Y_MAX)
        # self.desired_point_normalized.z = 0
        # print("Updating goal position to:", self.desired_point_normalized.x, self.desired_point_normalized.y)
        print("Updating goal position to:", self.desired_point_discretized.x, self.desired_point_discretized.y)

        # For Info Purposes
        self.cumulative_reward = 0.0

        # Set to false Done, because its calculated asyncronously
        self._episode_done = False

        odometry = self.get_odom()
        self.previous_distance_from_des_point = self.get_distance_from_desired_point(odometry.pose.pose.position)
        self.previous_heading_difference = self.calculate_heading_difference(odometry)

    def _init_env_variables(self):
        """
        Inits variables needed to be initialised each time we reset at the start
        of an episode.
        :return:
        """
        # For Info Purposes
        self.cumulative_reward = 0.0

        # Set to false Done, because its calculated asyncronously
        self._episode_done = False

        odometry = self.get_odom()
        self.previous_distance_from_des_point = self.get_distance_from_desired_point(odometry.pose.pose.position)
        self.previous_heading_difference = self.calculate_heading_difference(odometry)


    def _set_action(self, action):
        """
        Move the robot based on the action variable given
        """
        # We convert the actions to speed movements to send to the parent class AtomEnv
        if action == 0:  # FORWARD
            print("Action = 0, moving forward")
            linear_speed = self.linear_forward_speed
            angular_speed = 0.0
            self.last_action = "FORWARDS"
        elif action == 1:  # LEFT
            print("Action = 1, turning left")
            linear_speed = self.linear_turn_speed
            angular_speed = self.angular_speed
            self.last_action = "TURN_LEFT"
        elif action == 2:  # RIGHT
            print("Action = 2, turning right")
            linear_speed = self.linear_turn_speed
            angular_speed = -1 * self.angular_speed
            self.last_action = "TURN_RIGHT"

        # We tell the Atom the linear and angular speed to set to execute
        self.move_base(linear_speed, angular_speed, epsilon=0.05, update_rate=10)
    
    def _get_obs(self):
        """
        Here we define what sensor data of our robots observations
        To know which Variables we have acces to, we need to read the
        MyRobotEnv API DOCS
        :return: observations
        """
        rospy.logdebug("Start Get Observation ==>")
        # We get the laser scan data
        # laser_scan = self.get_laser_scan()

        # discretized_laser_scan = self.discretize_observation(laser_scan, self.new_ranges)

        # discretized_laser_scan_normalized = [round(min(item / self.max_laser_value, 1.0), 2) for item in discretized_laser_scan]
        # print("Laser scan normalized: " + str(discretized_laser_scan_normalized))

        # Retrieve the current Atom odom reading
        odometry = self.get_odom()
        x_position = odometry.pose.pose.position.x
        y_position = odometry.pose.pose.position.y
        odometry_original = [round(x_position / self.grid_size), round(y_position / self.grid_size)]

        odometry_array_discretized = self.discretizePoint(x_position, y_position, self.grid_size)
        # print("Odometry array: " + str(odometry_array_normalized))
        # print("Goal position: " + str(self.desired_point_discretized.x) + ","+ str(self.desired_point_discretized.y))

        relative_position_to_goal = self.calculate_relative_position(odometry_array_discretized[0], odometry_array_discretized[1],
                                                                     self.desired_point_discretized.x, self.desired_point_discretized.y)
        
        # relative_position_to_goal = self.calculate_relative_position(x_position, y_position, self.desired_point_discretized.x, self.desired_point_discretized.y)

        # normalizedState = self.normalize_state(x_position, y_position, relative_position_to_goal[0], relative_position_to_goal[1], ENV_X_MIN, 
        #                                        ENV_X_MAX, ENV_Y_MIN, ENV_Y_MAX)
        observations = odometry_array_discretized + relative_position_to_goal + odometry_original
        # observations = normalizedState + odometry_original

        return observations
    
    def _is_done(self, observations):
        """
        Decide if episode is done based on the observations
        """
        if self._episode_done:
            rospy.logerr("Atom is too close to wall, ending episode")
        else:
            current_position = Point()
            current_position.x = observations[0]
            current_position.y = observations[1]
            current_position.z = 0.0

            # We see if we are outside the Learning Space
            if current_position.x <= ENV_X_MAX and current_position.x > ENV_X_MIN:
                if current_position.y <= ENV_Y_MAX and current_position.y > ENV_Y_MIN:
                    rospy.logdebug("Current Atom Position is within the map" + str(current_position.x) + "," + str(
                        current_position.y) + "]")

                    #Check if we are at an obstacle
                    if self.is_near_obstacle(current_position):
                        rospy.logwarn("Atom near obstacle, ending episode")
                        self._episode_done = True

                    # We see if it got to the desired point
                    if self.is_in_desired_position(current_position):
                        rospy.logwarn("Atom has reached goal position!! Ending episode.")
                        self._episode_done = True
                else:
                    rospy.logerr("Atom outside of y bounds: " + str(current_position.y) + " ending episode")
                    self._episode_done = True
            else:
                rospy.logerr("Atom outside of x bounds: " + str(current_position.x) + " ending episode")
                self._episode_done = True

        return self._episode_done

    def _compute_reward(self, observations, done):
        """
        Return the reward based on the observations given
        """
        current_odometry = self.get_odom()

        current_position = Point()
        current_position.x = observations[0]
        current_position.y = observations[1]
        current_position.z = 0.0

        distance_from_des_point = self.get_distance_from_desired_point(current_position)
        distance_difference = distance_from_des_point - self.previous_distance_from_des_point

        current_heading_difference = self.calculate_heading_difference(current_odometry)
        heading_difference = current_heading_difference - self.previous_heading_difference

        reward = 0

        if not done:
            # If there has been a decrease in the distance to the desired point, we reward it
            if distance_difference < 0.0:
                rospy.logwarn("Rewarding decrease in distance to goal position")
                rospy.logwarn("Distance to goal position:" + str(distance_from_des_point))
                reward += self.forwards_reward
            else:
                rospy.logerr("Negative reward for no decrease in distance to goal position")
                rospy.logerr("Distance to goal position:" + str(distance_from_des_point))
                reward += -5

            if heading_difference < 0.0:
                rospy.logwarn("Rewarding decrease in heading difference")
                rospy.logwarn("Heading difference:" + str(heading_difference))

                reward += self.heading_reward
            else:
                rospy.logerr("Negative reward for no decrease in heading difference to goal position")
                rospy.logwarn("Heading difference:" + str(heading_difference))
                reward += -1
        #If the episode has ended, check if the robot is in the goal position or at an obstacle
        else:
            #If in the goal position, give large reward
            if self.is_in_desired_position(current_position):
                reward = self.end_episode_points
                rospy.logwarn("Reached goal position!!")

            if self.is_near_obstacle(current_position):
                reward = -1*self.end_episode_points
                rospy.logwarn("Atom ran into obstacle")
            #If not in the goal position and the episode ended something went wrong, penalize the actions
            else:
                reward = -1 * self.end_episode_points

        self.previous_distance_from_des_point = distance_from_des_point
        self.previous_heading_difference = heading_difference

        self.cumulative_reward += reward
        self.cumulative_steps += 1

        return reward

    def is_near_obstacle(self, current_position, epsilon=0.2):
        """
        It return True if the current position is similar to the desired poistion
        """
        
        # obstacle_1_x_pos_plus = self.normalize_feature(OBSTACLE_1_X + epsilon, ENV_X_MIN, ENV_X_MAX)
        # obstacle_1_x_pos_minus = self.normalize_feature(OBSTACLE_1_X - epsilon, ENV_X_MIN, ENV_X_MAX) 
        # obstacle_1_y_pos_plus = self.normalize_feature(OBSTACLE_1_Y + epsilon, ENV_X_MIN, ENV_X_MAX) 
        # obstacle_1_y_pos_minus = self.normalize_feature(OBSTACLE_1_Y - epsilon, ENV_X_MIN, ENV_X_MAX)
        # obstacle_2_x_pos_plus = self.normalize_feature(OBSTACLE_2_X + epsilon, ENV_X_MIN, ENV_X_MAX)
        # obstacle_2_x_pos_minus = self.normalize_feature(OBSTACLE_2_X - epsilon, ENV_X_MIN, ENV_X_MAX)
        # obstacle_2_y_pos_plus = self.normalize_feature(OBSTACLE_2_Y + epsilon, ENV_X_MIN, ENV_X_MAX)
        # obstacle_2_y_pos_minus = self.normalize_feature(OBSTACLE_2_Y - epsilon, ENV_X_MIN, ENV_X_MAX)
        
        # x_current = self.normalize_feature(current_position.x, ENV_X_MIN, ENV_X_MAX)
        # y_current = self.normalize_feature(current_position.y, ENV_X_MIN, ENV_X_MAX)

        obstacle_1_x_pos_plus = OBSTACLE_1_X + epsilon
        obstacle_1_x_pos_minus = OBSTACLE_1_X - epsilon
        obstacle_1_y_pos_plus = OBSTACLE_1_Y + epsilon
        obstacle_1_y_pos_minus = OBSTACLE_1_Y - epsilon
        obstacle_2_x_pos_plus = OBSTACLE_2_X + epsilon
        obstacle_2_x_pos_minus = OBSTACLE_2_X - epsilon
        obstacle_2_y_pos_plus = OBSTACLE_2_Y + epsilon
        obstacle_2_y_pos_minus = OBSTACLE_2_Y - epsilon
        
        x_current = current_position.x
        y_current = current_position.y

        obstacle_1_x_pos_are_close = (x_current <= obstacle_1_x_pos_plus) and (x_current > obstacle_1_x_pos_minus)
        obstacle_1_y_pos_are_close = (y_current <= obstacle_1_y_pos_plus) and (y_current > obstacle_1_y_pos_minus)
        obstacle_2_x_pos_are_close = (x_current <= obstacle_2_x_pos_plus) and (x_current > obstacle_2_x_pos_minus)
        obstacle_2_y_pos_are_close = (y_current <= obstacle_2_y_pos_plus) and (y_current > obstacle_2_y_pos_minus)

        at_obstacle = (obstacle_1_x_pos_are_close and obstacle_1_y_pos_are_close) or (obstacle_2_x_pos_are_close and obstacle_2_y_pos_are_close)

        return at_obstacle
    
    def is_in_desired_position(self, current_position, epsilon=0.05):
        """
        It return True if the current position is similar to the desired poistion
        """
        x_pos_plus = self.desired_point_normalized.x + epsilon
        x_pos_minus = self.desired_point_normalized.x - epsilon
        y_pos_plus = self.desired_point_normalized.y + epsilon
        y_pos_minus = self.desired_point_normalized.y - epsilon

        x_current = current_position.x
        y_current = current_position.y

        x_pos_are_close = (x_current <= x_pos_plus) and (x_current > x_pos_minus)
        y_pos_are_close = (y_current <= y_pos_plus) and (y_current > y_pos_minus)

        is_in_desired_pos = x_pos_are_close and y_pos_are_close

        return is_in_desired_pos
    
    def calculate_heading_difference(self, current_odometry):
        quaternion = (current_odometry.pose.pose.orientation.x, current_odometry.pose.pose.orientation.y,
              current_odometry.pose.pose.orientation.z, current_odometry.pose.pose.orientation.w)

        # Convert quaternion to Euler angles
        euler = tf.transformations.euler_from_quaternion(quaternion)

        # Yaw angle
        yaw = euler[2]

        # Robot's current position
        robot_x = current_odometry.pose.pose.position.x
        robot_y = current_odometry.pose.pose.position.y

        # Goal position
        goal_x = self.desired_point_discretized.x  
        goal_y = self.desired_point_discretized.y 

        # Calculate the angle from the robot to the goal
        angle_to_goal = math.atan2(goal_y - robot_y, goal_x - robot_x)

        # Calculate the difference in heading
        heading_difference = angle_to_goal - yaw

        # Normalize the angle to the range [-pi, pi]
        heading_difference = math.atan2(math.sin(heading_difference), math.cos(heading_difference))

        return heading_difference

    def calculate_relative_position(self, agent_x, agent_y, goal_x, goal_y):
        delta_x = goal_x - agent_x
        delta_y = goal_y - agent_y
        return [round(delta_x, 1), round(delta_y,1)]

    def get_distance_from_desired_point(self, current_position):
        """
        Calculates the distance from the current position to the desired point
        :param start_point:
        :return:
        """
        distance = self.get_distance_from_point(current_position,
                                                self.desired_point)

        return distance

    def get_distance_from_point(self, pstart, p_end):
        """
        Given a Vector3 Object, get distance from current position
        :param p_end:
        :return:
        """
        a = numpy.array((pstart.x, pstart.y, pstart.z))
        b = numpy.array((p_end.x, p_end.y, p_end.z))

        distance = numpy.linalg.norm(a - b)

        return distance

    def discretize_observation(self, data, new_ranges):
        """
        Discards all the laser readings that are not multiple in index of new_ranges
        value.
        """
        self._episode_done = False

        discretized_ranges = []
        mod = len(data.ranges) / new_ranges

        for i, item in enumerate(data.ranges):
            if (i % mod == 0):
                if item == float('Inf') or numpy.isinf(item):
                    discretized_ranges.append(self.max_laser_value)
                elif numpy.isnan(item):
                    discretized_ranges.append(self.min_laser_value)
                else:
                    discretized_ranges.append(int(item))

                if (self.min_range > item > 0):
                    self._episode_done = True

        return discretized_ranges
