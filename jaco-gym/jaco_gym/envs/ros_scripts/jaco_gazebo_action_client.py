#! /usr/bin/env python

import actionlib
import rospy
import numpy as np
import cv2
import math
import time
from ultralytics import YOLO

from control_msgs.msg import FollowJointTrajectoryAction, FollowJointTrajectoryGoal, JointTrajectoryControllerState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from gazebo_msgs.msg import LinkStates, ModelState
from std_srvs.srv import Empty
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Pose, Point
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from moveit_msgs.srv import GetStateValidityRequest, GetStateValidity



class JacoGazeboActionClient:

    def __init__(self):
        
        # action_address = "/j2n6s300/effort_joint_trajectory_controller/follow_joint_trajectory"
        action_address = "/j2n4s300/effort_joint_trajectory_controller/follow_joint_trajectory" #4 dof
        self.client = actionlib.SimpleActionClient(action_address, FollowJointTrajectoryAction)



        self.pub_topic = '/gazebo/set_model_state'
        self.pub = rospy.Publisher(self.pub_topic, ModelState, queue_size=1)

        # self.model = YOLO('/home/rl/YOLOv8/runs/segment/train6/weights/best.pt')
        # self.i = 0
        # Unpause the physics
        rospy.wait_for_service('/gazebo/unpause_physics')
        unpause_gazebo = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
        unpause_gazebo()
        


    def move_arm(self, points_list):
        # # Unpause the physics
        # rospy.wait_for_service('/gazebo/unpause_physics')
        # unpause_gazebo = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
        # unpause_gazebo()

        self.client.wait_for_server()

        goal = FollowJointTrajectoryGoal()    

        # We need to fill the goal message with its components
        #         
        # check msg structure with: rosmsg info FollowJointTrajectoryGoal
        # It is composed of 4 sub-messages:
        # "trajectory" of type trajectory_msgs/JointTrajectory 
        # "path_tolerance" of type control_msgs/JointTolerance
        # "goal_tolerance" of type control_msgs/JointTolerance
        # "goal_time_tolerance" of type duration

        trajectory_msg = JointTrajectory()
        # check msg structure with: rosmsg info JointTrajectory
        # It is composed of 3 sub-messages:
        # "header" of type std_msgs/Header 
        # "joint_names" of type string
        # "points" of type trajectory_msgs/JointTrajectoryPoint

        # trajectory_msg.joint_names = [
        #     "j2n6s300_joint_1", 
        #     "j2n6s300_joint_2", 
        #     "j2n6s300_joint_3", 
        #     "j2n6s300_joint_4", 
        #     "j2n6s300_joint_5", 
        #     "j2n6s300_joint_6"
        #     ]

        trajectory_msg.joint_names = [
            "j2n4s300_joint_1", 
            "j2n4s300_joint_2", 
            "j2n4s300_joint_3", 
            "j2n4s300_joint_4", 
            ]       #4 dof

        points_msg = JointTrajectoryPoint()
        # check msg structure with: rosmsg info JointTrajectoryPoint
        # It is composed of 5 sub-messages:
        # "positions" of type float64
        # "velocities" of type float64
        # "accelerations" of type float64
        # "efforts" of type float64
        # "time_from_start" of type duration
        
        points_msg.positions = points_list
        # points_msg.velocities = [0, 0, 0, 0, 0, 0]
        # points_msg.accelerations = [0, 0, 0, 0, 0, 0]
        # points_msg.effort = [0, 0, 0, 0, 0, 0]
        points_msg.velocities = [0, 0, 0, 0]    #4 dof
        points_msg.accelerations = [0, 0, 0, 0] #4 dof
        points_msg.effort = [0, 0, 0, 0]        #4 dof
        points_msg.time_from_start = rospy.Duration(0.01)

        # fill in points message of the trajectory message
        trajectory_msg.points = [points_msg]

        # fill in trajectory message of the goal
        goal.trajectory = trajectory_msg

        # self.client.send_goal_and_wait(goal)
        self.client.send_goal(goal)
        self.client.wait_for_result()

        rospy.sleep(2)      # wait for 2s

        # return self.client.get_state()

    def move_sphere(self, coords_list):

        model_state_msg = ModelState()
        # check msg structure with: rosmsg info gazebo_msgs/ModelState
        # It is composed of 4 sub-messages:
        # model_name of type string
        # pose of type geometry_msgs/Pose
        # twist of type geometry_msgs/Twist 
        # reference_frame of type string

        pose_msg = Pose()
        # rosmsg info geometry_msgs/Pose
        # It is composed of 2 sub-messages
        # position of type geometry_msgs/Point
        # orientation of type geometry_msgs/Quaternion 

        quat = self.euler_to_quaternion(coords_list[5], coords_list[4], coords_list[3])
        pose_msg.orientation.x = quat[0]
        pose_msg.orientation.y = quat[1]
        pose_msg.orientation.z = quat[2]
        pose_msg.orientation.w = quat[3]

        point_msg = Point()
        # rosmsg info geometry_msgs/Point
        # It is composed of 3 sub-messages
        # x of type float64
        # y of type float64
        # z of type float64
        point_msg.x = coords_list[0]
        point_msg.y = coords_list[1]
        point_msg.z = coords_list[2]

        pose_msg.position = point_msg

        model_state_msg.model_name = "my_sphere3"
        model_state_msg.pose = pose_msg
        model_state_msg.reference_frame = "world"

        # print(model_state_msg)
        
        self.pub.publish(model_state_msg)



    def cancel_move(self):
        self.client.cancel_all_goals()


    def read_state_old(self):
        # self.status = rospy.wait_for_message("/j2n6s300/effort_joint_trajectory_controller/state", JointTrajectoryControllerState)
        self.status = rospy.wait_for_message("/j2n4s300/effort_joint_trajectory_controller/state", JointTrajectoryControllerState)      #4 dof

        # convert tuple to list and concatenate
        self.state = list(self.status.actual.positions) + list(self.status.actual.velocities)
        # also self.status.actual.accelerations, self.status.actual.effort

        return self.state


    def read_state(self):
        # self.status = rospy.wait_for_message("/j2n6s300/joint_states", JointState)
        self.status = rospy.wait_for_message("/j2n4s300/joint_states", JointState)      #4 dof
        
        self.joint_names = self.status.name
        # print(self.joint_names)

        self.pos = self.status.position
        self.vel = self.status.velocity
        self.eff = self.status.effort

        # return self.status
        return np.asarray(self.pos + self.vel + self.eff)


    def read_state_simple(self):
        """
        read state of the joints only (not the finglers) + removed the efforts
        """

        # self.status = rospy.wait_for_message("/j2n6s300/joint_states", JointState)
        self.status = rospy.wait_for_message("/j2n4s300/joint_states", JointState)      #4 dof
        
        # self.joint_names = self.status.name[:6]
        self.joint_names = self.status.name[:4]     #4 dof
        # print(self.joint_names)

        # self.pos = self.status.position[:6]
        # self.vel = self.status.velocity[:6]
        self.pos = self.status.position[:4]     #4 dof
        self.vel = self.status.velocity[:4]     #4 dof

        # return self.status
        return np.asarray(self.pos + self.vel)

    def read_image(self):
        # time.sleep(5)
        # ---------------rgb_image---------------#
        self.status = rospy.wait_for_message("/camera_link/color/image_raw", Image)
        # print(self.status)
        self.status = CvBridge().imgmsg_to_cv2(self.status, "bgr8")
        # cv2.imwrite(str(self.i)+".jpg",self.status)
        # self.i+=1
        # self.status = cv2.cvtColor(self.status,cv2.COLOR_BGR2GRAY)
        # self.status = cv2.resize(self.status, (64, 64)) 
        # preprocessor = ImagePreprocessor(observation_shape=self.status.shape, grayscale=True)
        # self.status = preprocessor.transform(image)

        #-------------------delete background-------------------#
        # gray = cv2.cvtColor(self.status,cv2.COLOR_BGR2GRAY)
        # cv2.imwrite("test_gazebo_gray.jpg",gray)
        # a=np.where(gray>150)
        # self.status[a[0],a[1],:]=255
        # cv2.imwrite("test.jpg",self.status)
        image=self.status.copy()
        image[:,:,:]=255
        arm_pos=np.where(self.status[:,:,1]<140)
        obj_pos=np.where((self.status[:,:,1].astype(int)-self.status[:,:,0].astype(int))>30)
        image[arm_pos[0],arm_pos[1],:]=0
        image[obj_pos[0],obj_pos[1],:]=[0,255,0]
        self.status=image
        #-------------------delete background-------------------#
        #-------------------YOLO-------------------#
        # results = self.model(self.status)
        # image = self.status.copy()
        # image[:,:,:] = 255
        # self.status = results[0].plot(boxes=False,probs=False,img=image)
        #-------------------YOLO-------------------#
        return self.status
        # ---------------------------------------#
        # ---------------depth_image---------------#
        # self.status = rospy.wait_for_message("/camera_link/depth/image_raw", Image)
        # # print(self.status)
        # self.status = CvBridge().imgmsg_to_cv2(self.status, desired_encoding="8UC1")
        # # print(self.status)
        # image_norm = cv2.normalize(self.status,None,0,255,cv2.NORM_MINMAX)
        # # image = cv2.applyColorMap(image_norm,cv2.COLORMAP_JET)
        # return image_norm

    def get_object_pos(self):
        self.status = rospy.wait_for_message("/gazebo/model_states", LinkStates)
        self.pos = self.status.pose
        return [self.status.pose[2].position.x, self.status.pose[2].position.y, self.status.pose[2].position.z]

    def get_tip_coord(self):
        self.status = rospy.wait_for_message("/gazebo/link_states", LinkStates)
        # see also topic /tf

        self.joint_names = self.status.name
        self.pos = self.status.pose

        # BE CAREFUL: joint number changes if I add a sphere!
        # print(self.joint_names[8])
        # print(self.status.pose[8].position)


        # for i in range(14):
        #     print(i)
        #     print("joint:")
        #     print(self.joint_names[i])
        #     print("pose:")
        #     print(self.status.pose[i])

        return [(self.status.pose[8].position.x+self.status.pose[10].position.x)/2, (self.status.pose[8].position.y+self.status.pose[10].position.y)/2, (self.status.pose[8].position.z+self.status.pose[10].position.z)/2]
        
    def euler_to_quaternion(self, yaw, pitch, roll):

        qx = np.sin(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) - np.cos(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
        qy = np.cos(roll/2) * np.sin(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.cos(pitch/2) * np.sin(yaw/2)
        qz = np.cos(roll/2) * np.cos(pitch/2) * np.sin(yaw/2) - np.sin(roll/2) * np.sin(pitch/2) * np.cos(yaw/2)
        qw = np.cos(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)

        return [qx, qy, qz, qw]

    def quaternion_to_euler(self, w, x, y, z):
        ysqr = y * y
        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + ysqr)
        r = math.atan2(t0, t1)

        t2 = +2.0 * (w * y - z * x)
        t2 = +1.0 if t2 > +1.0 else t2
        t2 = -1.0 if t2 < -1.0 else t2
        p = math.asin(t2)

        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 -2.0 * (ysqr + z * z)
        y = math.atan2(t3, t4)
        return [r*180/np.pi, p*180/np.pi, y*180/np.pi]

    def get_endeffector_coord(self, arm_pos):
        x = 0.41 * np.cos(-arm_pos[0]) * np.cos(arm_pos[1] - np.pi/2) + 0.0098 * np.sin(-arm_pos[0]) + 0.3673 * np.cos(-arm_pos[0]) * np.sin(arm_pos[1]-arm_pos[2]-np.pi)
        y = 0.41 * np.sin(-arm_pos[0]) * np.cos(arm_pos[1] - np.pi/2) - 0.0098 * np.cos(-arm_pos[0]) + 0.3673 * np.sin(-arm_pos[0]) * np.sin(arm_pos[1]-arm_pos[2]-np.pi)
        z = 0.41 * np.sin(arm_pos[1] - np.pi/2) - 0.3673 * np.cos(arm_pos[2] - arm_pos[1] + np.pi) + 0.2755
        return [x, y, z]
# rospy.init_node("kinova_client")

# client = JacoGazeboActionClient()
# client.cancel_move()
# client.move_arm([3, 1.57, 3.14, 0, 0, 0])

# # client.move_sphere([1, 1, 1])

# print(client.read_state_simple())
# # print(client.read_state2())
# print(client.get_tip_coord())   # PB: reading coordinate doesn't wait until the arm has finished moving. SOLUTION: wait for 2s. To improve.

# client.read_state()
