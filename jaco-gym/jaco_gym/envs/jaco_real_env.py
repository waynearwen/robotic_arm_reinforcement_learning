import gym
import numpy as np
import random
import math
import cv2

from gym import error, spaces, utils
from gym.utils import seeding
from jaco_gym.envs.ros_scripts.jaco_real_action_client import JacoRealActionClient



class JacoEnv(gym.Env):

    def __init__(self):

        self.robot = JacoRealActionClient()
        self.i=0
        
    def step(self, action):
        # print(action)
        # execute action
        self.robot.cancel_move()
        action = action + self.pos
        self.robot.move_arm(
            action[0], 
            action[1], 
            action[2], 
            action[3], 
            # action[4],
            # action[5] 
        )
        self.pos = action
        self.reward = 0
        # self.state,self.center = self.robot.read_image()
        self.state = self.robot.read_image()
        gray = cv2.cvtColor(self.state,cv2.COLOR_BGR2GRAY)
        obj_pos=np.where(gray==150)
        # print(len(obj_pos[0]))
        if len(obj_pos[0]) == 0:
            print("None!!!!!!!")
            obj_pos = self.obj_pos_pre
        elif len(obj_pos[0]) > self.obj_pos_max:
            self.obj_pos_max = len(obj_pos[0])
        obj_mid_x=int((max(obj_pos[1])+min(obj_pos[1]))/2)
        obj_mid_y=int((max(obj_pos[0])+min(obj_pos[0]))/2)
        arm_pos=np.where(gray==0)
        arm_mid_x=int((max(arm_pos[1])+min(arm_pos[1]))/2)
        arm_mid_y=int((max(arm_pos[0])+min(arm_pos[0]))/2)
        white_pixel=np.where(gray[min(obj_pos[0])-50:max(obj_pos[0]),arm_mid_x]>240)
        # white_pixel=np.where(gray[min(obj_pos[0])-50:min(obj_pos[0]),obj_mid_x]>240)
        # d = math.sqrt((obj_mid_x - arm_mid_x)**2 + (obj_mid_y - arm_mid_y)**2)
        # test = self.state.copy()
        # test[0:240,arm_mid_x-30]=[0,255,0]
        # cv2.imshow("test",test)
        self.done = False
        if self.i > 10:
            # print(d)
            # print(len(obj_pos[0]))
            # print(len(arm_pos[0]))
            # print(len(arm_pos[0])/len(obj_pos[0]))
            # print(len(arm_pos[0])/self.obj_pos_max)
            print("up:"+str(((7665/self.obj_pos_max)-0.89)*1.51+5.5))
            print("low:"+str(((7665/self.obj_pos_max)-0.87)*1.51+4.5))
            print(len(arm_pos[0])/self.obj_pos_max)
            print("pixel:"+str(len(white_pixel[0])))
            # print(len(obj_pos[0])/8800*29)
            print(len(obj_pos[0])/8800*49)
            if len(arm_pos[0])/self.obj_pos_max < ((7665/self.obj_pos_max)-0.89)*1.51+5.5 and len(arm_pos[0])/self.obj_pos_max > ((7665/self.obj_pos_max)-0.87)*1.51+4.5 and len(white_pixel[0]) < len(obj_pos[0])/8800*45:# and len(white_pixel[0]) > len(obj_pos[0])/8800*29 :
                self.done = True
        self.info = 0
        # self.state = self.robot.read_angles()
        # print("action:",action)
        # print("real:",self.robot.read_angles())
        # print("error",self.robot.read_angles()-action)
        # self.center_pre = self.center
        # self.d_pre = self.d_now
        self.i+=1
        self.obj_pos_pre = obj_pos
        return self.state, self.reward, self.done, self.info


    def reset(self): 

        # for i in range(6):
        # for i in range(4):
        #     self.robot.cancel_move()
        #     # self.robot.move_arm(0, 180, 180, 0, 0, 0)
        #     self.robot.move_arm(0, 180, 180, 0)     #4 dof
        # action = [0,120,270,0]
        self.robot.cancel_move()
        self.robot.move_arm(270, 235, 70, 0)
        print("Jaco reset to initial position")
        # self.state, self.center = self.robot.read_image()
        self.state = self.robot.read_image()
        gray = cv2.cvtColor(self.state,cv2.COLOR_BGR2GRAY)
        obj_pos=np.where(gray==150)
        self.obj_pos_pre = obj_pos
        self.obj_pos_max = len(obj_pos[0])
        # self.d_pre = math.sqrt((self.center.data[0] - self.center.data[2])**2 + (self.center.data[1] - self.center.data[3])**2)
        # # get state
        # self.state = self.robot.read_angles()
        # print(self.robot.read_angles())
        # # generate random coordinates of a point in space
        # x_target = random.uniform(-0.49, 0.49)
        # y_target = random.uniform(-0.49, 0.49)
        # z_target = random.uniform(0.69, 1.18)
        
        # self.target_vect = np.array([x_target, y_target, z_target])
        self.pos = [270,235,70,0]
        # print(self.center_pre.data)
        return self.state

    def render(self, mode='human', close=False):
        pass
