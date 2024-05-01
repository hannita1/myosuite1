import numpy as np
import os
import myosuite
import gym
from stable_baselines3 import PPO

models_dir = "models/PPO"
logs_dir = "logs"

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

env = gym.make('MyoHandPhoneFixed-v0')
env.reset()


# mean episode reward is not changing, why? always -1
model = PPO('MlpPolicy', env, verbose=1, tensorboard_log=logs_dir)

TIMESTEPS = 10000
for i in range(1,300):
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name='PPO')
    #save model every 10000 steps
    model.save(f"{models_dir}/{TIMESTEPS*i}")

for step in range(1000):
    env.mj_render()
    obs, reward, done, info = env.step(env.action_space.sample())
    #print(reward)

env.close
