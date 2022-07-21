import rospy
import math
import copy
import random
import numpy as np
from shapely.geometry import Point
from simple_laserscan.msg import Spying
from gazebo_msgs.msg import ModelStates, ModelState
from geometry_msgs.msg import Twist
from std_srvs.srv import Empty
from gazebo_msgs.srv import SetModelState
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from dvs_msgs.msg import EventArray, Event


class GazeboEnvironment:
    """
    Class for Gazebo Environment

    Main Function:
        1. Reset: Rest environment at the end of each episode
        and generate new goal position for next episode

        2. Step: Execute new action and return state
     """
    def __init__(self,
                 dvs_dim=(6, 480, 640),
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
        self.goal_dis_amp = 5
        self.goal_dir_amp = 5
        self.obstacle_poly_list = None
        self.robot_init_pose_list = None
        self.obs_near_th = obs_near_th
        self.goal_near_th = goal_near_th
        self.goal_reward = goal_reward
        self.obs_reward = obs_reward
        self.step_time = step_time
        # cv_bridge class
        self.cv_bridge = CvBridge()
        # Robot State
        self.robot_pose = [0., 0., 0.]
        self.robot_speed = [0., 0.]
        ##self.robot_scan = np.zeros(self.scan_dir_num)
        self.events_cubic = np.zeros(dvs_dim)
        self.robot_state_init = False
        self.robot_depth_init = False
        # Goal Position
        self.goal_position = [0., 0.]
        self.goal_dis_dir_pre = [0., 0.]
        self.goal_dis_dir_cur = [0., 0.]
        # speed range
        self.linear_spd_range = 0.5
        self.angular_spd_range = 2.0
        # Subscriber
        print('pp')
        rospy.Subscriber('/gazebo/model_states', ModelStates, self._robot_state_cb)
        rospy.Subscriber('/mybot/camera1/events', EventArray, self._robot_dvs_cb)
        rospy.Subscriber('/Spying_signal', Spying, self._robot_spying_cb)
        print('hh')
        # Publisher
        self.pub_action = rospy.Publisher('/robot/cmd_vel', Twist, queue_size=5)
        # Service
        self.pause_gazebo = rospy.ServiceProxy('gazebo/pause_physics', Empty)
        self.unpause_gazebo = rospy.ServiceProxy('gazebo/unpause_physics', Empty)
        self.set_model_target = rospy.ServiceProxy('gazebo/set_model_state', SetModelState)
        self.reset_simulation = rospy.ServiceProxy('gazebo/reset_simulation', Empty)
        # Init Subscriber
        while not self.robot_state_init:
            print('loop1')
            continue
        ## while not self.robot_scan_init:
        ##     continue
        # while not self.robot_depth_init:
        #     print('loop2')
        #     continue
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
        state = self._robot_state_2_ddpg_state(next_rob_state)  # [np.array(1x4), list(np.array(6x480x640), np.array(6x480x640))]
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
        target_msg.model_name = 'target_robot'
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
        robot_msg.model_name = 'robot'
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
        state = self._robot_state_2_ddpg_state(rob_state)
        return state  # [list(1x4), np.array(6x480x640)] 

    def set_new_environment(self, init_pose_list, goal_list, obstacle_list):
        """
        Set New Environment for training
        :param init_pose_list: init pose list of robot
        :param goal_list: goal position list
        :param obstacle_list: obstacle list
        """
        self.robot_init_pose_list = [[3,3,0],[4,4,0],[3.5,4,0],[6.3,5,0],[6,6,0]]
        self.goal_pos_list = [[6,3],[7,4],[5.4,4],[9.4,5],[10,6]]
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

    def _get_next_robot_state(self):
        """
        Get the combination of state after execute the action for a certain time

        State will be: [robot_pose, robot_spd, scan]
        :return: state
        """
        tmp_robot_pose = copy.deepcopy(self.robot_pose)
        tmp_robot_spd = copy.deepcopy(self.robot_speed)
        tmp_robot_dvs = copy.deepcopy(self.events_cubic)
        state = [tmp_robot_pose, tmp_robot_spd, tmp_robot_dvs]
        """
        tmp_robot_pose : [x, y, yaw]
        tmp_robot_spd  : [sqrt(linear.x**2 + linear.y**2), angular.z]
        tmp_robot_depth: np.array(): (1 x 480 x 640)
        """
        return state

    def _robot_state_2_ddpg_state(self, state):
        """
        Transform robot state to DDPG state
        Robot State: [robot_pose, robot_spd, robot_dvs]
        DDPG state: [Distance to goal, Direction to goal, Linear Spd, Angular Spd, depth_img]
        :param state: robot state
        :return: ddpg_state
        """
        tmp_state = [0 for _ in range(4)]
        if state[1][0] > 0:
            tmp_state[0] = state[1][0] / self.linear_spd_range
        else:
            tmp_state[1] = abs(state[1][0]) / self.linear_spd_range
        if state[1][1] > 0:
            tmp_state[2] = state[1][1] / self.angular_spd_range
        else:
            tmp_state[3] = abs(state[1][1]) / self.angular_spd_range

        tmp_state = np.array(tmp_state)
        tmp_state = tmp_state[np.newaxis, :]
        ddpg_state = [tmp_state]   # [sqrt(x**2 + y**2), msg.twist[-1].angular.z]
        '''
        Transform distance in laser scan to [0, scale]
        '''
        positive_dvs = state[2][0]
        negative_dvs = state[2][1]
        rescale_dvs_pos = np.clip(positive_dvs, 0, 1)   # np.array(6x480x640)
        rescale_dvs_neg = np.clip(negative_dvs, 0, 1)
        rescale_dvs = [rescale_dvs_pos, rescale_dvs_neg]
        ddpg_state.append(rescale_dvs)
        return ddpg_state   # [np.array(1x4), list(np.array(6x480x640), np.array(6x480x640))]

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

    def _robot_spying_cb(self, msg):
        """
        Callback function for spying the state of robot and target
        :param msg: spying signal
        """
        self.goal_dis_dir_cur = [msg.distance, msg.direction]
        # print(self.goal_dis_dir_cur)

    def _robot_state_cb(self, msg):
        """
        Callback function for robot state
        :param msg: message
        """
        # print('------ robot_state_cb ------')
        if self.robot_state_init is False:
            self.robot_state_init = True
        quat = [msg.pose[-2].orientation.x,
                msg.pose[-2].orientation.y,
                msg.pose[-2].orientation.z,
                msg.pose[-2].orientation.w]
        siny_cosp = 2. * (quat[0] * quat[1] + quat[2] * quat[3])
        cosy_cosp = 1. - 2. * (quat[1] ** 2 + quat[2] ** 2)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        linear_spd = math.sqrt(msg.twist[-2].linear.x**2 + msg.twist[-2].linear.y**2)
        self.robot_pose = [msg.pose[-2].position.x, msg.pose[-2].position.y, yaw]      # 分别是 X, Y, rotation by Z
        self.robot_speed = [linear_spd, msg.twist[-2].angular.z]                       # [sqrt(linear.x**2, linear.y**2), angular.z]
        # print('robot pose :', self.robot_pose)
        # print('robot speed:', self.robot_speed)

    def _robot_dvs_cb(self, msg):
        """
        Callback function for robot dvs events
        :param msg: event msg
        """
        event_size = len(msg.events)   # the size of events
        width, height = msg.width, msg.height
        num1_pos_img = np.zeros((1, height, width))
        num1_neg_img = np.zeros((1, height, width))
        num2_pos_img = np.zeros((1, height, width))
        num2_neg_img = np.zeros((1, height, width))
        num3_pos_img = np.zeros((1, height, width))
        num3_neg_img = np.zeros((1, height, width))
        num4_pos_img = np.zeros((1, height, width))
        num4_neg_img = np.zeros((1, height, width))
        num5_pos_img = np.zeros((1, height, width))
        num5_neg_img = np.zeros((1, height, width))
        num6_pos_img = np.zeros((1, height, width))
        num6_neg_img = np.zeros((1, height, width))
        # events_cubic = np.zeros((6, height, width))
        position1 = int(event_size / 6)
        position2 = int(2*position1)
        position3 = int(3*position1)
        position4 = int(4*position1)
        position5 = int(5*position1)

        for i in range(event_size):
            x, y = msg.events[i].x, msg.events[i].y
            if i < position1:
                if msg.events[i].polarity:
                    num1_pos_img[0][y][x] = num1_pos_img[0][y][x] + 1
                else:
                    num1_neg_img[0][y][x] = num1_neg_img[0][y][x] + 1
            elif i > position1 and i < position2:
                if msg.events[i].polarity:
                    num2_pos_img[0][y][x] = num2_pos_img[0][y][x] + 1
                else:
                    num2_neg_img[0][y][x] = num2_neg_img[0][y][x] + 1
            elif i > position2 and i < position3:
                if msg.events[i].polarity:
                    num3_pos_img[0][y][x] = num3_pos_img[0][y][x] + 1
                else:
                    num3_neg_img[0][y][x] = num3_neg_img[0][y][x] + 1                       
            elif i > position3 and i < position4:
                if msg.events[i].polarity:
                    num4_pos_img[0][y][x] = num4_pos_img[0][y][x] + 1
                else:
                    num4_neg_img[0][y][x] = num4_pos_img[0][y][x] + 1
            elif i > position4 and i < position5:
                if msg.events[i].polarity:
                    num5_pos_img[0][y][x] = num5_pos_img[0][y][x] + 1
                else:
                    num5_neg_img[0][y][x] = num5_pos_img[0][y][x] + 1
            else:
                if msg.events[i].polarity:
                    num6_pos_img[0][y][x] = num6_pos_img[0][y][x] + 1
                else:
                    num6_neg_img[0][y][x] = num6_pos_img[0][y][x] + 1             

        positive_events = np.concatenate([num1_pos_img, num2_pos_img, num3_pos_img, num4_pos_img, num5_pos_img, num6_pos_img], axis=0)
        negative_events = np.concatenate([num1_neg_img, num2_neg_img, num3_neg_img, num4_neg_img, num5_neg_img, num6_neg_img], axis=0)

        self.events_cubic = [positive_events, negative_events]  # events_cubic: np.array(6 x 480 x 640)
        # print(self.events_cubic)
        # print('bb')