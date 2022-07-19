#!/usr/bin/env python3.8

import rospy
import numpy as np
import math
from gazebo_msgs.srv import *
from simple_laserscan.msg import Spying

def compute_dis_dir_2_goal(target, robot):
     """
     compute the difference of distance and direction to goal position
     :param target: the position of target
     :param robot : the position of robot
     :return : distance, direction
     """
     delta_x = target[0] - robot[0]
     delta_y = target[1] - robot[1]
     distance = math.sqrt(delta_x**2 + delta_y**2)
     ego_direction = math.atan2(delta_y, delta_x)
     robot_direction = robot[2]   # yaw
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
     print(distance)
     print(direction)
     return distance, direction  # direction 是有正有负的，根据右手定则规定正负


def process_pose(raw_pose):
     """
     :return : yaw
     """
     x, y, z, w = raw_pose
     siny_cosp = 2. * (x * y * z * w)
     cosy_cosp = 1. - 2. * (y**2 + z**2)
     yaw = math.atan2(siny_cosp, cosy_cosp)
     return yaw


if __name__ == '__main__':
     
     rospy.init_node('pub_dis_dir')
     get_model = rospy.ServiceProxy('gazebo/get_model_state', GetModelState)
     model = GetModelStateRequest()
     dis_dir_pub = rospy.Publisher('/Spying_signal', Spying, queue_size=1)

     while not rospy.is_shutdown():

          spying_data = Spying()
          model.model_name = 'target_robot'
          target_pos_msg = get_model(model)
          target_pos = [target_pos_msg.pose.position.x, target_pos_msg.pose.position.y]
          model.model_name = 'robot'
          mrobot_pos_msg = get_model(model)
          mrobot_pos = [mrobot_pos_msg.pose.position.x, mrobot_pos_msg.pose.position.y]
          mrobot_yaw = [mrobot_pos_msg.pose.orientation.x, mrobot_pos_msg.pose.orientation.y, \
                    mrobot_pos_msg.pose.orientation.z, mrobot_pos_msg.pose.orientation.w]
          yaw_rob = process_pose(mrobot_yaw)
          mrobot_pos.append(yaw_rob)

          dis, dir = compute_dis_dir_2_goal(target_pos, mrobot_pos)
          spying_data.distance = dis
          spying_data.direction = dir
          dis_dir_pub.publish(spying_data)
     
