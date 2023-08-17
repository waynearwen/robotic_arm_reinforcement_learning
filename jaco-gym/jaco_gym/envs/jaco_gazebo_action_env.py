# import gym
# import numpy as np
# import random
# import cv2
# from gym import error, spaces, utils
# from gym.utils import seeding
# from jaco_gym.envs.ros_scripts.jaco_gazebo_action_client import JacoGazeboActionClient



# class JacoEnv(gym.Env):

#     def __init__(self):

#         self.robot = JacoGazeboActionClient()

#         self.action_dim = 4     #4 dof
#         self.timestep = 1

#         high = np.ones([self.action_dim])
#         self.action_space = gym.spaces.Box(-high, high)
        
#         self.observation_space = gym.spaces.Box(low=0, high=255, shape=(480,640,3), dtype=np.uint8)  #rgb_image
#         # self.observation_space = gym.spaces.Box(low=0, high=255, shape=(480,640), dtype=np.float)   #depth_image

#         self.reward_list = []
#         self.timestep_list = []
#         self.step_list = []
#         self.reward = 0
#         self.count = 1
#         random.seed(127)
        

#     def convert_action_to_deg(self, a, OldMin, OldMax, NewMin, NewMax):
    
#         OldRange = (OldMax - OldMin)  
#         NewRange = (NewMax - NewMin)  

#         return (((a - OldMin) * NewRange) / OldRange) + NewMin
    
    
#     def action2deg(self, action):
#         action[0] = self.convert_action_to_deg(action[0], OldMin=-1, OldMax=1, NewMin=245, NewMax=295)
#         action[1] = self.convert_action_to_deg(action[1], OldMin=-1, OldMax=1, NewMin=225, NewMax=250)
#         action[2] = self.convert_action_to_deg(action[2], OldMin=-1, OldMax=1, NewMin=70, NewMax=90)
#         # action[3] = self.convert_action_to_deg(action[3], OldMin=-1, OldMax=1, NewMin=0, NewMax=360)
#         action[3] = 180
#         return action

#     def step(self, action):

#         # self.pos = [270, 180, 90, 180]
#         # self.pos = np.radians(self.pos)
#         # self.robot.move_arm(self.pos)
        
#         # self.action = self.action2deg(action) 
#         # self.action = np.radians(self.action)
#         self.action = 0.08*action + self.pos
#         if self.action[0] > 295 * np.pi/180:
#             self.action[0] = 295 * np.pi/180
#         elif self.action[0] < 245 * np.pi/180:
#             self.action[0] = 245 * np.pi/180
#         if self.action[1] > 250 * np.pi/180:
#             self.action[1] = 250 * np.pi/180
#         elif self.action[1] < 225 * np.pi/180:
#             self.action[1] = 225 * np.pi/180
#         if self.action[2] > 90 * np.pi/180:
#             self.action[2] = 90 * np.pi/180
#         elif self.action[2] < 70 * np.pi/180:
#             self.action[2] = 70 * np.pi/180
#         self.robot.move_arm(self.action)

#         self.observation = self.robot.read_image()

#         self.done = False

#         self.endeffector_coord = self.robot.get_endeffector_coord(self.action)
#         self.dist_to_target = np.linalg.norm(self.endeffector_coord - self.target_vect[0:3])

#         # arm_vector = self.rotate_vector([1,0,0],self.action[1]-self.action[2]-np.pi/2,self.action[3]-np.pi,np.pi*3/2-self.action[0],"arm")
#         # target_vector = self.rotate_vector([0,0,1],self.target_vect[3],self.target_vect[4],self.target_vect[5],"target")
#         # cal_dot = abs(np.dot(arm_vector,target_vector))

#         # if self.dist_to_target < 0.06:
#         #     self.done = True
#         #     self.reward += 4
#         #     print("----------Done---------")
#         #     if self.count == 1:
#         #         self.reward += 1
#         #         print("------------1-----------")
#         #     # self.reward = 1 + 0.5*(1-cal_dot)
#         # else:
#         #     self.reward -= self.dist_to_target - 0.06
#         #     if self.dist_to_target == self.dist_to_target_pre:
#         #         self.reward -= 0.5
#         #     # self.reward = 0
#         #     # if self.dist_to_target < self.dist_to_target_pre:
#         #     #     self.reward += abs(self.dist_to_target - self.dist_to_target_pre)
#         #     #     # self.reward += 0.02
#         #     # elif self.dist_to_target > self.dist_to_target_pre:
#         #     #     self.reward -= abs(self.dist_to_target - self.dist_to_target_pre)
#         #     # else:
#         #     #     self.reward -= 0.5

#         # if cal_dot < 0.2:
#         #     self.reward += 0.2
#         #     print("-----correct-----")
#         # else:
#         #     self.reward -= 0.2*cal_dot

#         # ------------------------------test-------------------------------#
#         # if self.dist_to_target < 0.06:
#         #     self.done = True
#         #     # self.reward = 1
#         #     print("----------Done---------")
#         #     # self.reward = 1 + 0.5*(1-cal_dot)
#         #     self.reward = 2
#         # else:
#         #     self.reward = -self.dist_to_target 
#         #     # self.reward = 0
#         #     # if self.dist_to_target < self.dist_to_target_pre:
#         #     #     self.reward += abs(self.dist_to_target - self.dist_to_target_pre)
#         #     #     # self.reward += 0.02
#         #     # elif self.dist_to_target > self.dist_to_target_pre:
#         #     #     self.reward -= abs(self.dist_to_target - self.dist_to_target_pre)
#         #     # else:
#         #     #     self.reward -= 0.5
#         # ------------------------------test-------------------------------#

#         # ------------------------------lstm7-----------------------------------#
#         if self.dist_to_target < 0.06:
#             self.done = True
#             self.reward = 1
#             print("----------Done---------")
#             # self.reward = 1 + 0.5*(1-cal_dot)
#             # self.reward = 2
#         else:
#             self.reward = -self.dist_to_target 
#             # self.reward = 0
#             # if self.dist_to_target < self.dist_to_target_pre:
#             #     self.reward += abs(self.dist_to_target - self.dist_to_target_pre)
#             #     # self.reward += 0.02
#             # elif self.dist_to_target > self.dist_to_target_pre:
#             #     self.reward -= abs(self.dist_to_target - self.dist_to_target_pre) + 0.05
#             # else:
#             #     self.reward -= 0.05
#         # ------------------------------lstm7-----------------------------------#

#         # create info
#         self.info = {"endeffector coordinates": self.endeffector_coord, "image": self.observation}

#         print("epsoide",self.count,"   Reward:" + str(self.reward))
#         print("distance:",self.dist_to_target)
#         # print("cal_dot:",cal_dot)

#         # self.each_reward = self.reward - self.reward_pre
#         self.each_reward = self.reward
#         self.timestep_list.append(self.each_reward)
#         np.save("reward_timestep.npy",self.timestep_list)

#         self.count += 1
#         self.timestep += 1
#         self.reward_pre = self.reward
#         self.dist_to_target_pre = self.dist_to_target
#         self.pos = self.action

#         return self.observation, self.reward, self.done, self.info

#     def rotation_matrix(self, roll, pitch, yaw, who):
#         R_x = np.array([[1, 0, 0],
#                         [0, np.cos(roll), -np.sin(roll)],
#                         [0, np.sin(roll), np.cos(roll)]])
#         R_y = np.array([[np.cos(pitch), 0, np.sin(pitch)],
#                         [0, 1, 0],
#                         [-np.sin(pitch), 0, np.cos(pitch)]])
#         R_z = np.array([[np.cos(yaw), -np.sin(yaw), 0],
#                         [np.sin(yaw), np.cos(yaw), 0],
#                         [0, 0, 1]])

#         if who == "target":
#             return np.dot(np.dot(R_z, R_y), R_x)
#         elif who == "arm":
#             return np.dot(np.dot(R_z, R_x), R_y)

#     def rotate_vector(self, vec, roll, pitch, yaw, who):
#         R = self.rotation_matrix(roll, pitch, yaw, who)
#         return np.dot(R, vec)

#     def reset(self): 

#         self.robot.cancel_move()

#         self.step_list.append(self.count-1)
#         np.save("step.npy",self.step_list)
#         self.reward_list.append(self.reward)
#         np.save("reward.npy",self.reward_list)
#         self.reward = 0
#         self.reward_pre = 0
#         # self.each_reward = 0
#         self.count = 1

#         # self.pos = [270, 180, 90, 180]
#         self.pos = [270, 235, 70, 180]
#         self.pos = np.radians(self.pos)
#         self.robot.move_arm(self.pos)
#         print("Jaco reset to initial position")

#         self.obs = self.robot.read_image()

#         x_target = random.uniform(-0.1539, 0.1717)
#         # x_target = random.uniform(-0.08, 0.08)
#         # y_target = random.uniform(-0.4472, -0.4589)
#         y_target = -0.4472
#         z_target = random.uniform(0.0706, 0.2515)
#         roll_target = random.uniform(0, 2*np.pi)    #radius
#         pitch_target = random.uniform(0, 2*np.pi)
#         yaw_target = random.uniform(0, 2*np.pi)
#         self.target_vect = np.array([x_target, y_target, z_target, roll_target, pitch_target, yaw_target])
#         print("Random target pose generated")
#         self.robot.move_sphere(self.target_vect)

#         self.endeffector_coord = self.robot.get_endeffector_coord(self.pos) #radians
#         self.dist_to_target_pre = np.linalg.norm(self.endeffector_coord - self.target_vect[0:3])

#         return self.obs


#     def render(self, mode='human', close=False):
#         pass


#----------------------------------------------------------RL-----------------------------------------------------#
import gym
import numpy as np
import random
import cv2
import time
from gym import error, spaces, utils
from gym.utils import seeding
from jaco_gym.envs.ros_scripts.jaco_gazebo_action_client import JacoGazeboActionClient



class JacoEnv(gym.Env):

    def __init__(self):

        self.robot = JacoGazeboActionClient()

        # self.action_dim = 6
        self.action_dim = 4    #4 dof
        # self.obs_dim = 36
        # self.obs_dim = 12   # when using read_state_simple
        # self.obs_dim = 8    #4 dof
        self.timestep = 0

        high = np.ones([self.action_dim])
        self.action_space = gym.spaces.Box(np.float32(-high), np.float32(high))
        
        # high = np.inf * np.ones([self.obs_dim])
        # self.observation_space = gym.spaces.Box(-high, high)
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(480,640,3), dtype=np.uint8)
        
        # self.observation_space = gym.spaces.Dict(
        #     {
        #         "image":gym.spaces.Box(low=0, high=255, shape=(64,64,3), dtype=np.uint8)
        #         "joint":gym.spaces.Box(low=0, high=2*np.pi, shape=(4,), dtype=np.float32)
        #     })

        self.reward_list = []
        self.timestep_list = []
        self.step_list = []
        self.reward = 0
        self.count = 1
        self.fail = 0
        random.seed(127)

    def rotation_matrix(self, roll, pitch, yaw, who):
        R_x = np.array([[1, 0, 0],
                        [0, np.cos(roll), -np.sin(roll)],
                        [0, np.sin(roll), np.cos(roll)]])
        R_y = np.array([[np.cos(pitch), 0, np.sin(pitch)],
                        [0, 1, 0],
                        [-np.sin(pitch), 0, np.cos(pitch)]])
        R_z = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                        [np.sin(yaw), np.cos(yaw), 0],
                        [0, 0, 1]])

        if who == "target":
            return np.dot(np.dot(R_z, R_y), R_x)
        elif who == "arm":
            return np.dot(np.dot(R_z, R_x), R_y)

    def rotate_vector(self, vec, roll, pitch, yaw, who):
        R = self.rotation_matrix(roll, pitch, yaw, who)
        return np.dot(R, vec)

    def step(self, action):
        self.count += 1

        # pos = [0, 180, 270, 0]      #4 dof
        # pos = np.radians(pos)
        # self.robot.move_arm(pos)
        # self.action = self.action2deg_delta(action)
        # self.action = np.radians(self.action)
        # self.robot.move_arm(self.action + pos)
        
        # self.action = action*0.02 + self.action
        self.action = np.radians(action) + self.action
        self.robot.move_arm(self.action)

        # if self.target_vect[1] < 0.2:
        #     self.target_vect[1] += 0.005
        # self.robot.move_sphere(self.target_vect)

        # # print(action)
        # # convert action from range [-1, 1] to [0, 360] 
        # self.action = self.action2deg(action)
        # # print(self.action)
        # # convert to radians    
        # self.action = np.radians(self.action)

        # # move arm 
        # self.robot.move_arm(self.action)
       
        # get state
        # self.observation = self.robot.read_state()
        # self.observation = self.robot.read_state_simple()   # only return 12 values instead of 36
        # time.sleep(1)
        self.observation = self.robot.read_image()
        # cv2.imwrite("state space.jpg",self.observation)
        self.done = False
        # calculate reward
        self.tip_coord = self.robot.get_endeffector_coord(self.action)
        # self.tip_coord = self.robot.get_tip_coord()
        self.dist_to_target = np.linalg.norm(self.tip_coord - self.target_vect[0:3])

        arm_vector = self.rotate_vector([1,0,0],self.action[1]-self.action[2]-np.pi/2,self.action[3]-np.pi,np.pi*3/2-self.action[0],"arm")
        target_vector = self.rotate_vector([0,0,1],self.target_vect[3],self.target_vect[4],self.target_vect[5],"target")
        cal_dot = abs(np.dot(arm_vector,target_vector))

        object_pos = self.robot.get_object_pos()

        self.reward = 0

        ##Reward Funtion ################################################
        # if object_pos!=self.object_pos_pre:
        #     self.done = True
        #     self.reward = -2
        #     print("----------collision-----------")
        # elif self.dist_to_target < 0.06:
        if self.dist_to_target < 0.06:
        # if self.dist_to_target < 0.035:
            self.done = True
            # self.reward = 2 + (1 - cal_dot)
            self.reward = 2
            print("----------Done---------")
        else:
        #     # self.reward = 1 - self.dist_to_target 
        #     self.reward = - self.dist_to_target 
            # if self.tip_coord[1] > (self.tip_coord[0] - 0.06) * np.tan(np.radians(33.8)) or self.tip_coord[1] < -(self.tip_coord[0] - 0.06) * np.tan(np.radians(33.8)) or self.tip_coord[2] < 0.045 or self.tip_coord[2] > (self.tip_coord[0] - 0.06) * np.tan(np.radians(22.8)) + 0.1:
            if self.tip_coord[0] > (-self.tip_coord[1] - 0.11) * np.tan(np.radians(33.8)) or self.tip_coord[0] < -(-self.tip_coord[1] - 0.11) * np.tan(np.radians(33.8)) or self.tip_coord[2] < 0.045 or self.tip_coord[2] > (-self.tip_coord[1] - 0.11) * np.tan(np.radians(22.8)) + 0.145:
                # self.reward -= 10
                self.reward -= 5
                print(self.tip_coord)
                # print((-self.tip_coord[1] - 0.11) * np.tan(np.radians(33.8)))
                # print(-(-self.tip_coord[1] - 0.11) * np.tan(np.radians(33.8)))
                print("----------Fail---------")
                # cv2.imwrite(str(self.fail)+".jpg",self.observation)
                self.done = True 
                self.fail += 1
            else:
                if self.dist_to_target < self.dist_to_target_pre:
                    self.reward += abs(self.dist_to_target - self.dist_to_target_pre) * 10
                else:
                    self.reward -= abs(self.dist_to_target - self.dist_to_target_pre) * 10
        ##Reward funtion END #######################################################
        self.dist_to_target_pre = self.dist_to_target
        # create info
        self.info = {"tip coordinates": self.tip_coord, "target coordinates": self.target_vect}

        # create done
        # self.done = False

        # IF DEFINING DONE AS FOLLOWS, THE EPISODE ENDS EARLY AND A GOOD AGENT WILL RECEIVED A PENALTY FOR BEING GOOD
        # COOMENT THIS
        # if self.dist_to_target < 0.01:
            # self.done = True

        # print("tip position: ", self.tip_coord)
        # print("target vect: ", self.target_vect)
        print("epsoide",self.count,"   Reward:"+str(self.reward))
        print("dist_to_target: ", self.dist_to_target)
        self.each_reward = self.reward
        self.timestep_list.append(self.each_reward)
        np.save("reward_timestep.npy",self.timestep_list)
        return self.observation, self.reward, self.done, self.info


    def reset(self): 

        self.robot.cancel_move()
        self.step_list.append(self.count-1)
        np.save("step.npy",self.step_list)
        self.reward_list.append(self.reward)
        np.save("reward.npy",self.reward_list)
        self.reward = 0
        # self.each_reward = 0
        self.count = 1

        # pos = [0, 180, 180, 0, 0, 0]
        # pos = [0, 120, 270, 0]      #4 dof
        pos = [270, 240, 90, 0] 
        pos[0] = random.uniform(250, 290)
        pos[1] = random.uniform(230, 245)
        pos[2] = random.uniform(80, 100)
        pos[3] = random.uniform(0, 360)
        # pos = [0,180,180,0]  
        pos = np.radians(pos)
        self.robot.move_arm(pos)
        print("Jaco reset to initial position")
        self.count = 0
        # get observation
        # self.obs = self.robot.read_state()
        # self.obs = self.robot.read_state_simple()

        # generate random coordinates of a point in space
        # limits of real robot
        # x_target = random.uniform(-0.49, 0.49)
        # y_target = random.uniform(-0.49, 0.49)
        # z_target = random.uniform(0.69, 1.18)

        # limits in Gazebo
        # x_target = random.uniform(0.48, 0.62)
        # y_target = random.uniform(-0.18, 0.18)
        # z_target = random.uniform(0.05, 0.2)
        y_target = random.uniform(-0.48, -0.62)
        x_target = random.uniform(-0.18, 0.18)
        z_target = random.uniform(0.05, 0.2)
        roll_target = random.uniform(0, 2*np.pi)    #radius
        pitch_target = random.uniform(0, 2*np.pi)
        yaw_target = random.uniform(0, 2*np.pi)

        # x_target = -0.18
        # y_target = -0.18
        # # z_target = random.uniform(0.05, 0.12)
        # z_target = 0.05

        self.target_vect = np.array([x_target, y_target, z_target, roll_target, pitch_target, yaw_target])
        self.object_pos_pre=[x_target, y_target, z_target]
        # self.target_vect = np.array([x_target, y_target, z_target])
        print("Random target coordinates generated")

        # if testing: graphically move the sphere target, if training, comment this line
        self.robot.move_sphere(self.target_vect)

        self.tip_coord = self.robot.get_endeffector_coord(pos)
        # self.tip_coord = self.robot.get_tip_coord()
        # print(self.tip_coord)
        # self.dist_to_target_pre = np.linalg.norm(self.tip_coord - self.target_vect)
        self.dist_to_target_pre = np.linalg.norm(self.tip_coord - self.target_vect[0:3])
        # print(self.tip_coord)
        self.action = pos
        # time.sleep(5)
        self.obs = self.robot.read_image()
        return self.obs


    def render(self, mode='human', close=False):
        pass
