import gym
import jaco_gym
import random
import rospy
import cv2
# from stable_baselines3 import PPO

rospy.init_node("kinova_client", anonymous=True, log_level=rospy.INFO)

env = gym.make('JacoReal-v0')
# model = PPO.load("/home/rl/results/5/JacoGazebo-v1(10000)")

state = env.reset()
cv2.imwrite("observation_real.jpg",state)

# action = [0,120,270,0]
# state = env.step(action)
# env.print_tip_pos()
# print("current state: ", state)


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
