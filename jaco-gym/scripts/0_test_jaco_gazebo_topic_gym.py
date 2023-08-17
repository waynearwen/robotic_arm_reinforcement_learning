import gym
import jaco_gym
import random
import numpy as np 
import rospy
import cv2
from stable_baselines3.common.env_checker import check_env

rospy.init_node("kinova_client", anonymous=True, log_level=rospy.INFO)

env = gym.make('JacoGazebo-v1')
# print(env.observation_space)
# check_env(env)
state = env.reset()
# print(state)
# print(state.shape)
# print(np.isnan(state)
# cv2.imshow("observation",state)
# cv2.imwrite("observation.jpg",state)

# action = [-1.0,0.0,0.0,0.0]
# state = env.step(action)

# print("current state: ", state)

# env.print_tip_pos()



# for t in range(3):

#     # create action
#     ang0 = 0
#     ang1 = 180
#     ang2 = random.randrange(90, 270)
#     ang3 = random.randrange(0, 359)
#     ang4 = random.randrange(0, 359)
#     ang5 = random.randrange(0, 359)
#     action = [ang0, ang1, ang2, ang3, ang4, ang5]
#     print("action sent: ", action)

#     state = env.step(action)
#     print("current state: ", state)


#     print("time step {}".format(t))

env.close()