import rospy
import math
import copy
import random
import numpy as np
from shapely.geometry import Point
from simple_laserscan.msg import SimpleScan
from gazebo_msgs.msg import ModelStates, ModelState
from geometry_msgs.msg import Twist
from std_srvs.srv import Empty
from gazebo_msgs.srv import SetModelState
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError


class GazeboEnvironment:
    """
    Class for Gazebo Environment

    Main Function:
        1. Reset: Rest environment at the end of each episode
        and generate new goal position for next episode

        2. Step: Execute new action and return state
     """
    def __init__(self,
                 depth_scale = 1.0,
                 depth_min_dis = 0.05,
                 depth_img_dim = (48, 64),
                 goal_dis_min_dis=0.5,
                 goal_dis_scale=1.0,
                 obs_near_th=0.35,
                 goal_near_th=0.5,
                 goal_reward=10,
                 obs_reward=-5,
                 goal_dis_amp=5,
                 goal_dir_amp=5,
                 step_time=0.1):
        """

        :param laser_scan_half_num: half number of scan points
        :param laser_scan_min_dis: Min laser scan distance
        :param laser_scan_scale: laser scan scale
        :param scan_dir_num: number of directions in laser scan
        :param goal_dis_min_dis: minimal distance of goal distance
        :param goal_dis_scale: goal distance scale
        :param obs_near_th: Threshold for near an obstacle
        :param goal_near_th: Threshold for near an goal
        :param goal_reward: reward for reaching goal
        :param obs_reward: reward for reaching obstacle
        :param goal_dis_amp: amplifier for goal distance change
        :param step_time: time for a single step (DEFAULT: 0.1 seconds)
        """
        self.goal_pos_list = None
        self.obstacle_poly_list = None
        self.robot_init_pose_list = None
        self.depth_scale = depth_scale
        self.depth_min_dis = depth_min_dis
        self.depth_img_dim = depth_img_dim
        self.goal_dis_min_dis = goal_dis_min_dis
        self.goal_dis_scale = goal_dis_scale
        self.obs_near_th = obs_near_th
        self.goal_near_th = goal_near_th
        self.goal_reward = goal_reward
        self.obs_reward = obs_reward
        self.goal_dis_amp = goal_dis_amp
        self.goal_dir_amp = goal_dir_amp
        self.step_time = step_time
        # cv_bridge class
        self.cv_bridge = CvBridge()
        # Robot State
        self.robot_pose = [0., 0., 0.]
        self.robot_speed = [0., 0.]
        ##self.robot_scan = np.zeros(self.scan_dir_num)
        self.robot_depth_img = np.zeros(self.depth_img_dim)
        self.robot_state_init = False
        ##self.robot_scan_init = False
        self.robot_depth_init = False
        # Goal Position
        self.goal_position = [0., 0.]
        self.goal_dis_dir_pre = [0., 0.]  # Last step goal distance and direction
        self.goal_dis_dir_cur = [0., 0.]  # Current step goal distance and direction
        # Subscriber
        rospy.Subscriber('gazebo/model_states', ModelStates, self._robot_state_cb)
        ##rospy.Subscriber('/mrobot/simplescan', SimpleScan, self._robot_scan_cb)
        rospy.Subscriber('/kinect/depth/image_raw', Image, self._robot_depth_cb)
        # Publisher
        self.pub_action = rospy.Publisher('/cmd_vel', Twist, queue_size=5)
        # Service
        self.pause_gazebo = rospy.ServiceProxy('gazebo/pause_physics', Empty)
        self.unpause_gazebo = rospy.ServiceProxy('gazebo/unpause_physics', Empty)
        self.set_model_target = rospy.ServiceProxy('gazebo/set_model_state', SetModelState)
        self.reset_simulation = rospy.ServiceProxy('gazebo/reset_simulation', Empty)
        # Init Subscriber
        while not self.robot_state_init:
            continue
        ## while not self.robot_scan_init:
        ##     continue
        while not self.robot_depth_init:
            continue
        rospy.loginfo("Finish Subscriber Init...")

    def step(self, action):
        """
        Step Function for the Environment

        Take a action for the robot and return the updated state
        :param action: action taken
        :return: state, reward, done
        """
        assert self.goal_pos_list is not None
        assert self.obstacle_poly_list is not None
        rospy.wait_for_service('gazebo/unpause_physics')
        try:
            self.unpause_gazebo()
        except rospy.ServiceException as e:
            print("Unpause Service Failed: %s" % e)
        '''
        First give action to robot and let robot execute and get next state
        '''
        move_cmd = Twist()
        move_cmd.linear.x = action[0]
        move_cmd.angular.z = action[1]
        self.pub_action.publish(move_cmd)
        rospy.sleep(self.step_time)
        next_rob_state = self._get_next_robot_state()
        rospy.wait_for_service('gazebo/pause_physics')
        try:
            self.pause_gazebo()
        except rospy.ServiceException as e:
            print("Pause Service Failed: %s" % e)
        '''
        Then stop the simulation
        1. Transform Robot State to DDPG State
        2. Compute Reward of the action
        3. Compute if the episode is ended
        '''
        goal_dis, goal_dir = self._compute_dis_dir_2_goal(next_rob_state[0])
        self.goal_dis_dir_cur = [goal_dis, goal_dir]
        state = self._robot_state_2_ddpg_state(next_rob_state)
        reward, done = self._compute_reward(next_rob_state)
        self.goal_dis_dir_pre = [self.goal_dis_dir_cur[0], self.goal_dis_dir_cur[1]]
        return state, reward, done

    def reset(self, ita):
        """
        Reset Function to reset simulation at start of each episode

        Return the initial state after reset
        :param ita: number of route to reset to
        :return: state
        """
        assert self.goal_pos_list is not None
        assert self.obstacle_poly_list is not None
        assert self.robot_init_pose_list is not None
        ita = ita % 100
        assert ita < len(self.goal_pos_list)
        rospy.wait_for_service('gazebo/unpause_physics')
        try:
            self.unpause_gazebo()
        except rospy.ServiceException as e:
            print("Unpause Service Failed: %s" % e)
        '''
        First choose new goal position and set target model to goal
        '''
        self.goal_position = self.goal_pos_list[ita]
        target_msg = ModelState()
        target_msg.model_name = 'target'
        target_msg.pose.position.x = self.goal_position[0]
        target_msg.pose.position.y = self.goal_position[1]
        rospy.wait_for_service('gazebo/set_model_state')
        try:
            resp = self.set_model_target(target_msg)
        except rospy.ServiceException as e:
            print("Set Target Service Failed: %s" % e)
        '''
        Then reset robot state and get initial state
        '''
        self.pub_action.publish(Twist())
        robot_init_pose = self.robot_init_pose_list[ita]
        robot_init_quat = self._euler_2_quat(yaw=robot_init_pose[2])
        robot_msg = ModelState()
        # robot_msg.model_name = 'mobile_base'
        robot_msg.model_name = 'mrobot'
        robot_msg.pose.position.x = robot_init_pose[0]
        robot_msg.pose.position.y = robot_init_pose[1]
        robot_msg.pose.orientation.x = robot_init_quat[1]
        robot_msg.pose.orientation.y = robot_init_quat[2]
        robot_msg.pose.orientation.z = robot_init_quat[3]
        robot_msg.pose.orientation.w = robot_init_quat[0]
        rospy.wait_for_service('gazebo/set_model_state')
        try:
            resp = self.set_model_target(robot_msg)
        except rospy.ServiceException as e:
            print("Set Target Service Failed: %s" % e)
        rospy.sleep(0.5)
        '''
        Transfer the initial robot state to the state for the DDPG Agent
        '''
        rob_state = self._get_next_robot_state()       # [tmp_robot_pose, tmp_robot_spd, tmp_robot_depth_img]
                                                       # :param tmp_robot_pose: [x, y, yaw]
                                                       # :param tmp_robot_spd : [sqrt(linear.x**2 + linear.y**2), angular.z]
                                                       # :param tmp_robot_depth_img: np.array() -- (480 x 640)
        rospy.wait_for_service('gazebo/pause_physics')
        try:
            self.pause_gazebo()
        except rospy.ServiceException as e:
            print("Pause Service Failed: %s" % e)
        goal_dis, goal_dir = self._compute_dis_dir_2_goal(rob_state[0])  # rob_state[0]: [x, y, rotation by z]
        self.goal_dis_dir_pre = [goal_dis, goal_dir]
        self.goal_dis_dir_cur = [goal_dis, goal_dir]
        state = self._robot_state_2_ddpg_state(rob_state)
        return state  # [list(1x4), np.array(1x48x64)] 

    def set_new_environment(self, init_pose_list, goal_list, obstacle_list):
        """
        Set New Environment for training
        :param init_pose_list: init pose list of robot
        :param goal_list: goal position list
        :param obstacle_list: obstacle list
        """
        self.robot_init_pose_list = init_pose_list
        self.goal_pos_list = goal_list
        self.obstacle_poly_list = obstacle_list

    def _euler_2_quat(self, yaw=0, pitch=0, roll=0):
        """
        Transform euler angule to quaternion
        :param yaw: z
        :param pitch: y
        :param roll: x
        :return: quaternion
        """
        cy = math.cos(yaw * 0.5)
        sy = math.sin(yaw * 0.5)
        cp = math.cos(pitch * 0.5)
        sp = math.sin(pitch * 0.5)
        cr = math.cos(roll * 0.5)
        sr = math.sin(roll * 0.5)
        w = cy * cp * cr + sy * sp * sr
        x = cy * cp * sr - sy * sp * cr
        y = sy * cp * sr + cy * sp * cr
        z = sy * cp * cr - cy * sp * sr
        return [w, x, y, z]

    def _compute_dis_dir_2_goal(self, pose):
        """
        Compute the difference of distance and direction to goal position
        :param pose: pose of the robot
        :return: distance, direction
        """
        delta_x = self.goal_position[0] - pose[0]
        delta_y = self.goal_position[1] - pose[1]
        distance = math.sqrt(delta_x**2 + delta_y**2)
        ego_direction = math.atan2(delta_y, delta_x)
        robot_direction = pose[2]   # yaw
        while robot_direction < 0:
            robot_direction += 2 * math.pi
        while robot_direction > 2 * math.pi:
            robot_direction -= 2 * math.pi
        while ego_direction < 0:
            ego_direction += 2 * math.pi
        while ego_direction > 2 * math.pi:
            ego_direction -= 2 * math.pi
        pos_dir = abs(ego_direction - robot_direction)
        neg_dir = 2 * math.pi - abs(ego_direction - robot_direction)
        if pos_dir <= neg_dir:
            direction = math.copysign(pos_dir, ego_direction - robot_direction)
        else:
            direction = math.copysign(neg_dir, -(ego_direction - robot_direction))
        return distance, direction  # direction 是有正有负的，根据右手定则规定正负

    def _get_next_robot_state(self):
        """
        Get the combination of state after execute the action for a certain time

        State will be: [robot_pose, robot_spd, scan]
        :return: state
        """
        tmp_robot_pose = copy.deepcopy(self.robot_pose)
        tmp_robot_spd = copy.deepcopy(self.robot_speed)
        tmp_robot_depth = copy.deepcopy(self.robot_depth_img)
        state = [tmp_robot_pose, tmp_robot_spd, tmp_robot_depth]
        """
        tmp_robot_pose : [x, y, yaw]
        tmp_robot_spd  : [sqrt(linear.x**2 + linear.y**2), angular.z]
        tmp_robot_depth: np.array(): (1 x 480 x 640)
        """
        return state

    def _robot_state_2_ddpg_state(self, state):
        """
        Transform robot state to DDPG state
        ## Robot State: [robot_pose, robot_spd, scan]
        Robot State: [robot_pose, robot_spd, robot_depth]
        DDPG state: [Distance to goal, Direction to goal, Linear Spd, Angular Spd, depth_img]
        :param state: robot state
        :return: ddpg_state
        """
        tmp_goal_dis = self.goal_dis_dir_cur[0]  # goal_dis
        if tmp_goal_dis == 0:
            tmp_goal_dis = self.goal_dis_scale   # 1
        else:
            tmp_goal_dis = self.goal_dis_min_dis / tmp_goal_dis  # 0.5 / tmp_goal_dis 
            if tmp_goal_dis > 1:
                tmp_goal_dis = 1
            tmp_goal_dis = tmp_goal_dis * self.goal_dis_scale    # tmp_goal_dis * 1
        ddpg_state = [[self.goal_dis_dir_cur[1], tmp_goal_dis, state[1][0], state[1][1]]]   # [Direction to goal, Distance to goal, sqrt(x**2 + y**2), msg.twist[-1].angular.z]
        '''
        Transform distance in laser scan to [0, scale]
        '''
        rescale_depth_img = self.depth_scale * (self.depth_min_dis / state[2])
        rescale_depth_img = np.clip(rescale_depth_img, 0, self.depth_scale)
        ddpg_state.append(rescale_depth_img)
        return ddpg_state   # [list(1x4), np.array(1x48x64)]


    def _compute_reward(self, state):   # state: robot state
        """
        Compute Reward of the action base on current robot state and last step goal distance and direction

        Reward:
            1. R_Arrive If Distance to Goal is smaller than D_goal
            2. R_Collision If Distance to Obstacle is smaller than D_obs
            3. a * (Last step distance to goal - current step distance to goal)

        If robot near obstacle then done
        :param state: DDPG state
        :return: reward, done
        """
        done = False
        '''
        First compute distance to all obstacles
        '''
        near_obstacle = False
        robot_point = Point(state[0][0], state[0][1])
        for poly in self.obstacle_poly_list:
            tmp_dis = robot_point.distance(poly)
            if tmp_dis < self.obs_near_th:
                near_obstacle = True
                break
        '''
        Assign Rewards
        '''
        if self.goal_dis_dir_cur[0] < self.goal_near_th:
            reward = self.goal_reward  # self.goal_reward = 10
            done = True
        elif near_obstacle:            # 如果跟障碍物离得太近就停止，并给予惩罚
            reward = self.obs_reward   # self.obs_reward = -5
            done = True
        else:                          # 如果越来越近，就是正的奖励，越远就给负的奖励
            reward = self.goal_dis_amp * (2**(self.goal_dis_dir_pre[0] - self.goal_dis_dir_cur[0]) - 1)  # self.goal_dis_amp = 5   5 * (distance_delta)
            reward = reward + self.goal_dir_amp * (abs(self.goal_dis_dir_pre[1]) - abs(self.goal_dis_dir_cur[1]))
        return reward, done

    def _robot_state_cb(self, msg):
        """
        Callback function for robot state
        :param msg: message
        """
        if self.robot_state_init is False:
            self.robot_state_init = True
        quat = [msg.pose[-1].orientation.x,
                msg.pose[-1].orientation.y,
                msg.pose[-1].orientation.z,
                msg.pose[-1].orientation.w]
        siny_cosp = 2. * (quat[0] * quat[1] + quat[2] * quat[3])
        cosy_cosp = 1. - 2. * (quat[1] ** 2 + quat[2] ** 2)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        linear_spd = math.sqrt(msg.twist[-1].linear.x**2 + msg.twist[-1].linear.y**2)
        self.robot_pose = [msg.pose[-1].position.x, msg.pose[-1].position.y, yaw]      # 分别是 X, Y, rotation by Z
        self.robot_speed = [linear_spd, msg.twist[-1].angular.z]                       # [sqrt(linear.x**2, linear.y**2), angular.z]

    def _robot_scan_cb(self, msg):
        """
        Callback function for robot scan
        :param msg: message
        """
        if self.robot_scan_init is False:
            self.robot_scan_init = True
        self.robot_scan = np.array(msg.data)   # np.array(data * 36)

    def _robot_depth_cb(self, msg):
        """
        Callback function for robot depth image
        :param msg: depth msg
        """
        if self.robot_depth_init is False:
            self.robot_depth_init = True
        tmp_depth_img = self.cv_bridge.imgmsg_to_cv2(msg, "16UC1")     # from msg to cv_img (480 x 640)
        np.save("./tmp_data.npy", tmp_depth_img)
        #print("real data: ", tmp_depth_img)
        tmp_data = np.zeros((self.depth_img_dim))
        for i in range(0, 480, 10):
            for j in range(0, 640, 10):
                if tmp_depth_img[i][j] == 0:
                    tmp_data[int(i/10)][int(j/10)] = 25
                else:
                    tmp_data[int(i/10)][int(j/10)] = tmp_depth_img[i][j]
        self.robot_depth_img = tmp_data[np.newaxis, :]   # change into (1 x 480 x 640)
