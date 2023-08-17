import gym
import jaco_gym
import os
import rospy

# from stable_baselines3.common.policies import CnnLstmPolicy, CnnPolicy, MlpPolicy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
# from stable_baselines import PPO2
from stable_baselines3 import PPO

rospy.init_node("kinova_client", anonymous=True, log_level=rospy.INFO)

env_id = 'JacoGazebo-v1'

timesteps = '(10000)'
# Create log dir
log_dir = "../results/"+env_id+"/"
os.makedirs(log_dir, exist_ok=True)

env = gym.make(env_id)
env = Monitor(env, filename=log_dir, allow_early_resets=True)
env = DummyVecEnv([lambda: env])

# env.reset()

# model = PPO2(MlpPolicy, env, verbose=1, n_steps=20)
# model = PPO2("CnnLstmPolicy", env, verbose=1, n_steps=20)
# model = PPO("CnnPolicy", env, verbose=1,n_steps=2,batch_size=2)
model = PPO("CnnPolicy", env, verbose=1)
# model = PPO("MultiInputPolicy", env, verbose=1, n_steps=128,policy_kwargs=dict(normalize_images=False))
# model = model.load("/home/rl/results/2/JacoGazebo-v1(10000)",env=env)
model.learn(total_timesteps=150000)

model.save(log_dir+env_id+timesteps)
env.reset()

env.close()
