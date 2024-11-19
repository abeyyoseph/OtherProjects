import numpy
import rospy
from openai_ros import robot_gazebo_env
from std_msgs.msg import Float64
from sensor_msgs.msg import JointState
from sensor_msgs.msg import Image
from sensor_msgs.msg import LaserScan
from sensor_msgs.msg import PointCloud2
from sensor_msgs.msg import Imu
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
import tf

class AtomEnv(robot_gazebo_env.RobotGazeboEnv):
    """Superclass for all CubeSingleDisk environments.
    """

    def __init__(self):
        """
        Initializes a new AtomEnv environment.
        Atom doesnt use controller_manager, therefore we wont reset the 
        controllers in the standard fashion. For the moment we wont reset them.
        
        To check any topic we need to have the simulations running, we need to do two things:
        1) Unpause the simulation: without that th stream of data doesnt flow. This is for simulations
        that are pause for whatever the reason
        2) If the simulation was running already for some reason, we need to reset the controlers.
        This has to do with the fact that some plugins with tf, dont understand the reset of the simulation
        and need to be reseted to work properly.
        
        The Sensors: The sensors accesible are the ones considered usefull for AI learning.
        
        Sensor Topic List:
        * /odom : Odometry readings of the Base of the Robot
        * /imu: Inertial Mesuring Unit that gives relative accelerations and orientations.
        * /scan: Laser Readings
        
        Actuators Topic List: /cmd_vel, 
        
        Args:
        """
        print("Start AtomEnv INIT...")
        
        self.controllers_list = []

        self.robot_name_space = "/wheeled_robot/atom"

        # We launch the init function of the Parent Class robot_gazebo_env.RobotGazeboEnv
        super(AtomEnv, self).__init__(controllers_list=self.controllers_list,
                                            robot_name_space=self.robot_name_space,
                                            reset_controls=False,
                                            start_init_physics_parameters=False)

        self.gazebo.unpauseSim()

        print("Beginning to check if all sensors are ready")
        self._check_all_sensors_ready()

        # We Start all the ROS related Subscribers and publishers
        print("Subscribing to odom and lidar topics")
        rospy.Subscriber(self.robot_name_space+"/odom", Odometry, self._odom_callback)
        # rospy.Subscriber("/imu", Imu, self._imu_callback)
        rospy.Subscriber("/scan", LaserScan, self._laser_scan_callback)

        print("Successfully subscribed to odom and lidar topics")

        self._cmd_vel_pub = rospy.Publisher(self.robot_name_space+'/cmd_vel', Twist, queue_size=1)

        self._check_publishers_connection()

        self.gazebo.pauseSim()
        
        print("Finished AtomEnv INIT...")

    # Methods needed by the RobotGazeboEnv
    # ----------------------------
    
    def _check_all_systems_ready(self):
        """
        Checks that all the sensors, publishers and other simulation systems are
        operational.
        """
        self._check_all_sensors_ready()
        return True


    # CubeSingleDiskEnv virtual methods
    # ----------------------------

    def _check_all_sensors_ready(self):
        rospy.loginfo("CHECKING ALL SENSORS READY")
        self._check_odom_ready()
        self._check_laser_scan_ready()
        rospy.loginfo("ALL SENSORS READY")

    def _check_odom_ready(self):
        self.odom = None
        rospy.loginfo("Waiting for odom to be READY...")
        while self.odom is None and not rospy.is_shutdown():
            try:
                self.odom = rospy.wait_for_message(self.robot_name_space+"/odom", Odometry, timeout=5.0)
                rospy.loginfo("Current " + self.robot_name_space + "/odom READY=>")

            except:
                rospy.loginfo("Current "+ self.robot_name_space + "/odom not ready yet, retrying..")

        return self.odom

    def _check_laser_scan_ready(self):
        self.laser_scan = None
        rospy.loginfo("Waiting for /scan to be READY...")
        while self.laser_scan is None and not rospy.is_shutdown():
            try:
                self.laser_scan = rospy.wait_for_message("/scan", LaserScan, timeout=1.0)
                rospy.loginfo("Current /scan READY=>")

            except:
                rospy.loginfo("Current /scan not ready yet, retrying for getting laser_scan")
        return self.laser_scan
        

    def _odom_callback(self, data):
        self.odom = data
    
    def _imu_callback(self, data):
        self.imu = data

    def _laser_scan_callback(self, data):
        self.laser_scan = data

        
    def _check_publishers_connection(self):
        """
        Checks if the cmd_vel publishers is working
        :return:
        """
        rate = rospy.Rate(10)  # 10hz
        while self._cmd_vel_pub.get_num_connections() == 0 and not rospy.is_shutdown():
            rospy.loginfo("No susbribers to _cmd_vel_pub yet so we wait and try again")
            try:
                rate.sleep()
            except rospy.ROSInterruptException:
                # This is to avoid error when world is rested, time when backwards.
                pass
        rospy.loginfo("_cmd_vel_pub Publisher Connected")

        rospy.loginfo("All Publishers READY")
    
    # Methods that the TrainingEnvironment will need to define here as virtual
    # because they will be used in RobotGazeboEnv GrandParentClass and defined in the
    # TrainingEnvironment.
    # ----------------------------
    def _set_init_pose(self):
        """Sets the Robot in its init pose
        """
        raise NotImplementedError()
    
    def _init_env_variables(self):
        """Inits variables needed to be initialised each time we reset at the start
        of an episode.
        """
        raise NotImplementedError()

    def _compute_reward(self, observations, done):
        """Calculates the reward to give based on the observations given.
        """
        raise NotImplementedError()

    def _set_action(self, action):
        """Applies the given action to the simulation.
        """
        raise NotImplementedError()

    def _get_obs(self):
        raise NotImplementedError()

    def _is_done(self, observations):
        """Checks if episode done based on observations given.
        """
        raise NotImplementedError()
    
    def quaternion_to_euler(self, quaternion):
        """
        Convert a quaternion to Euler angles (roll, pitch, yaw)
        """
        q = quaternion
        euler = tf.transformations.euler_from_quaternion([q.x, q.y, q.z, q.w])
        return euler  # Returns a tuple (roll, pitch, yaw)

    def has_turned_30_degrees(self, start_yaw, current_yaw):
        """
        Check if the robot has turned at least 30 degrees from the start_yaw to current_yaw.
        Both yaw values should be in radians.
        """
        yaw_change = current_yaw - start_yaw
        yaw_change_deg = abs(yaw_change * (180.0 / numpy.pi))  # Convert radians to degrees
        return yaw_change_deg


    # Methods that the TrainingEnvironment will need.
    # ----------------------------
    def move_base(self, linear_speed, angular_speed, epsilon=0.05, update_rate=10):
        """
        It will move the base based on the linear and angular speeds given.
        It will wait untill those twists are achived reading from the odometry topic.
        :param linear_speed: Speed in the X axis of the robot base frame
        :param angular_speed: Speed of the angular turning of the robot base frame
        :param epsilon: Acceptable difference between the speed asked and the odometry readings
        :param update_rate: Rate at which we check the odometry.
        :return: 
        """
        cmd_vel_value = Twist()
        cmd_vel_value.linear.x = linear_speed
        cmd_vel_value.angular.z = angular_speed
        self._check_publishers_connection()
        self._cmd_vel_pub.publish(cmd_vel_value)
        self.wait_until_twist_achieved(cmd_vel_value,
                                        epsilon,
                                        update_rate)
    
        

    def wait_until_twist_achieved(self, cmd_vel_value, epsilon, update_rate):
        """
        We wait for the cmd_vel twist given to be reached by the robot reading
        from the odometry.
        :param cmd_vel_value: Twist we want to wait to reach.
        :param epsilon: Error acceptable in odometry readings.
        :param update_rate: Rate at which we check the odometry.
        :return:
        """
        print("START wait_until_twist_achieved...")
        
        rate = rospy.Rate(update_rate)
        start_wait_time = rospy.get_rostime().to_sec()
        end_wait_time = 0.0
        epsilon = 0.05
                
        linear_speed = cmd_vel_value.linear.x
        angular_speed = cmd_vel_value.angular.z
        
        linear_speed_plus = linear_speed + epsilon
        linear_speed_minus = linear_speed - epsilon
        
        maxWaitAttempts = 20
        attemptCount = 0

        initial_odometry = self._check_odom_ready()

        # Assuming you have access to the robot's initial and current odometry data as `initial_odom` and `current_odom`
        start_orientation = self.quaternion_to_euler(initial_odometry.pose.pose.orientation)

        #Check if we are performing a turn manuever. If so, enforce 25 degree turn to enhance exploration
        if abs(angular_speed) > 0:
            while not rospy.is_shutdown():
                current_odometry = self._check_odom_ready()
                current_orientation = self.quaternion_to_euler(current_odometry.pose.pose.orientation)

                attemptCount += 1
                
                currentYawChange = self.has_turned_30_degrees(start_orientation[2], current_orientation[2])

                if currentYawChange >= 30:  # Comparing yaw angles
                    print("Achieved Desired 30 degree Change in Yaw!")
                    end_wait_time = rospy.get_rostime().to_sec()
                    break
                
                if attemptCount == maxWaitAttempts:
                    print("Couldnt reach desired yaw change after 25 attempts, continuing w yaw change: ", currentYawChange)
                    break
                rate.sleep()
        else:

            while not rospy.is_shutdown():
                current_odometry = self._check_odom_ready()

                attemptCount += 1
                odom_linear_vel = current_odometry.twist.twist.linear.x
                
                linear_vel_are_close = (odom_linear_vel <= linear_speed_plus) and (odom_linear_vel > linear_speed_minus)
                
                if linear_vel_are_close:
                    print("Achieved Desired Velocity!")
                    end_wait_time = rospy.get_rostime().to_sec()
                    break
                
                print("Not there yet, keep waiting...")
                if attemptCount == maxWaitAttempts:
                    print("Couldnt reach desired velocity after 10 attempts, continuing with lin vel: " + str(odom_linear_vel))
                    break
                rate.sleep()
                
        delta_time = end_wait_time- start_wait_time
                
        return delta_time
        

    def get_odom(self):
        return self.odom
        
    def get_imu(self):
        return self.imu
        
    def get_laser_scan(self):
        return self.laser_scan
