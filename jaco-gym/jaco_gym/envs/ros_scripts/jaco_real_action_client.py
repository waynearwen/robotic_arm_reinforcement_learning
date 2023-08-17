#! /usr/bin/env python

import rospy
import actionlib

from kinova_msgs.msg import ArmJointAnglesGoal, ArmJointAnglesAction
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from std_msgs.msg import Int32MultiArray


class JacoRealActionClient:

  def __init__(self):

    # self.topic_name = '/j2n6s300_driver/joints_action/joint_angles'
    self.topic_name = '/j2n4s300_driver/joints_action/joint_angles'   #4 dof
    self.client = actionlib.SimpleActionClient(self.topic_name, ArmJointAnglesAction)

  def move_arm(self, angle1, angle2, angle3, angle4):
    self.client.wait_for_server()
    goal = ArmJointAnglesGoal()
    goal.angles.joint1 = angle1
    goal.angles.joint2 = angle2
    goal.angles.joint3 = angle3
    goal.angles.joint4 = angle4
    self.client.send_goal(goal)
    self.client.wait_for_result()   #4 dof
    
  def read_image(self):
    self.status = rospy.wait_for_message("/web_cam", Image)
    self.status = CvBridge().imgmsg_to_cv2(self.status, "bgr8")
    # self.center = rospy.wait_for_message("/arm_center", Int32MultiArray)
    # print("aa")
    # cv2.imwrite("observation_real.jpg",self.status)
    # return self.status, self.center
    return self.status

  def cancel_move(self):
    self.client.cancel_all_goals()

  def read_angles(self):

    self.client.wait_for_server()
    self.client.wait_for_result()
    joint_angles = self.client.get_result()
    angles_list = [
      joint_angles.angles.joint1, 
      joint_angles.angles.joint2, 
      joint_angles.angles.joint3, 
      joint_angles.angles.joint4, 
      # joint_angles.angles.joint5, 
      # joint_angles.angles.joint6, 
    ]
    return angles_list

  # def read_tip_position(self):
  #   # pos = rospy.wait_for_message('/j2n6s300_driver/out/tool_pose', PoseStamped)
  #   pos = rospy.wait_for_message('/j2n4s300_driver/out/tool_pose', PoseStamped)   #4 dof
  #   pos_list = [pos.pose.position.x, pos.pose.position.y, pos.pose.position.z]
  #   return pos_list




# rospy.init_node('jaco_real_action_client_node')
# robot_client = JacoRealActionClient()

# robot_client.cancel_move()
# robot_client.move_arm(0, 180, 180, 0, 0, 0)
