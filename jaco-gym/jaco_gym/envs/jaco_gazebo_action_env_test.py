import gym
import numpy as np
import random
import cv2
import time
import rospy
from gym import error, spaces, utils
from gym.utils import seeding
from jaco_gym.envs.ros_scripts.jaco_gazebo_action_client import JacoGazeboActionClient

class JacoEnv(gym.Env):

    def __init__(self):

        self.robot = JacoGazeboActionClient()

        self.action_dim = 4    #4 dof
        self.timestep = 0

        high = np.ones([self.action_dim])
        self.action_space = gym.spaces.Box(np.float32(-high), np.float32(high))
        
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(480,640,3), dtype=np.uint8)

        self.reward_list = []
        self.timestep_list = []
        self.step_list = []
        self.reward = 0
        self.count = 1
        self.fail = 0
        random.seed(18)


    def step(self, action):
        self.count += 1
        self.timestep += 1

        self.time_now = rospy.Time.now().to_sec()
        # print("Time:",self.time_now-self.time_pre)

        self.action = action*0.02 + self.action
        self.robot.move_arm(self.action)

        self.observation = self.robot.read_image()
        # cv2.imwrite(str(self.timestep)+".jpg",self.observation)
        self.done = False

        self.tip_coord = self.robot.get_endeffector_coord(self.action)
        self.dist_to_target = np.linalg.norm(self.tip_coord - self.target_vect)

        #-------------calculate reward-------------#
        self.reward = 0
        if self.dist_to_target < 0.06:
            self.done = True
            self.reward = 2
            print("----------Done---------")
        else:
        #     # self.reward = 1 - self.dist_to_target 
        #     self.reward = - self.dist_to_target 
            if self.tip_coord[1] > (self.tip_coord[0] - 0.06) * np.tan(np.radians(33.8)) or self.tip_coord[1] < -(self.tip_coord[0] - 0.06) * np.tan(np.radians(33.8)) or self.tip_coord[2] < 0.045 or self.tip_coord[2] > (self.tip_coord[0] - 0.06) * np.tan(np.radians(22.8)) + 0.1:
                # self.reward -= 10
                self.reward -= 5
                print(self.tip_coord)
                print("----------Fail---------")
                # cv2.imwrite(str(self.fail)+".jpg",self.observation)
                self.done = True 
                self.fail += 1
            else:
                if self.dist_to_target < self.dist_to_target_pre:
                    self.reward += abs(self.dist_to_target - self.dist_to_target_pre) * 10
                else:
                    self.reward -= abs(self.dist_to_target - self.dist_to_target_pre) * 10
        #------------------------------------------#

        self.dist_to_target_pre = self.dist_to_target
        self.info = {"tip coordinates": self.tip_coord, "target coordinates": self.target_vect}

        # -------------move object-------------#
        # if self.target_vect[1] < 0.2:
        #     self.target_vect[1] -= 0.003
        # self.robot.move_sphere(self.target_vect)
        # -------------------------------------#

        print("epsoide",self.count,"   Reward:"+str(self.reward))
        print("dist_to_target: ", self.dist_to_target)
        self.each_reward = self.reward
        self.timestep_list.append(self.each_reward)
        np.save("reward_timestep.npy",self.timestep_list)
        self.time_pre = rospy.Time.now().to_sec()
        return self.observation, self.reward, self.done, self.info


    def reset(self): 

        self.robot.cancel_move()
        self.step_list.append(self.count-1)
        np.save("step.npy",self.step_list)
        self.reward_list.append(self.reward)
        np.save("reward.npy",self.reward_list)
        self.reward = 0
        self.count = 1

        # pos = [0, 180, 180, 0, 0, 0]
        pos = [0, 120, 270, 0]      #4 dof
        # pos = [0,180,180,0]  
        # pos = [-20, 120, 270, 0]  
        # pos = [270, 240, 90, 0] 
        pos = np.radians(pos)
        self.robot.move_arm(pos)
        print("Jaco reset to initial position")
        self.count = 0

        x_target = random.uniform(0.48, 0.62)
        y_target = random.uniform(-0.18, 0.18)
        z_target = random.uniform(0.05, 0.2)
        # x_target = random.uniform(-0.18, 0.18)
        # y_target = 0.13
        # z_target = random.uniform(0.05, 0.2)
        self.target_vect = np.array([x_target, y_target, z_target])
        self.robot.move_sphere(self.target_vect)
        print("Random target coordinates generated")

        self.tip_coord = self.robot.get_endeffector_coord(pos)
        self.dist_to_target_pre = np.linalg.norm(self.tip_coord - self.target_vect)
        self.action = pos
        time.sleep(0.1)
        self.obs = self.robot.read_image()
        self.time_pre = rospy.Time.now().to_sec()
        # print(self.time_pre)
        return self.obs


    def render(self, mode='human', close=False):
        pass
