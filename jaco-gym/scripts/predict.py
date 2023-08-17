import gym
import jaco_gym
import os
import rospy

# from stable_baselines.common.policies import CnnLstmPolicy, CnnPolicy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

rospy.init_node("kinova_client", anonymous=True, log_level=rospy.INFO)

env_id = 'JacoGazebo-v1'

# Create log dir
log_dir = "../results/"+env_id+"/"
os.makedirs(log_dir, exist_ok=True)

env = gym.make(env_id)
env = Monitor(env, filename=log_dir, allow_early_resets=True)
env = DummyVecEnv([lambda: env])

model = PPO.load("/home/rl/results/10/JacoGazebo-v1(10000)")

mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=5)

# Enjoy trained agent
# obs = env.reset()
# for i in range(100):
#     action, _states = model.predict(obs)
#     obs, rewards, dones, info = env.step(action)
#     env.render()