import gym
import jaco_gym
import os
import rospy
from stable_baselines3 import PPO

rospy.init_node("kinova_client", anonymous=True, log_level=rospy.INFO)

env = gym.make('JacoReal-v0')
# Create log dir
print("load model")
model = PPO.load("/home/rl/results/10/JacoGazebo-v1(10000)")
print("load finish")
# mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=1)

# Enjoy trained agent
obs = env.reset()
print("reset finish")
for i in range(1000):
    print("step:",i)
    action, _states = model.predict(obs)
    obs, rewards, dones, info= env.step(action)
    if dones:
        break
    env.render()
print("done!!!")
# obs = env.reset()
# print("reset")