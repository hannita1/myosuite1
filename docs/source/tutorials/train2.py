import numpy as np
import os
import myosuite
from myosuite.utils import gym
from stable_baselines3 import PPO

models_dir = "models/PPO"
logs_dir = "logs"

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

env = gym.make('myoHandPoseFixed-v0')
env.reset()

# mean episode reward is not changing, why? always -1
model = PPO('MlpPolicy', env, verbose=1, tensorboard_log=logs_dir)

TIMESTEPS = 10000
#for i in range(1,30):
#    model.learn(total_timesteps=TIMESTEPS)#, reset_num_timesteps=False, tb_log_name='PPO')
    #save model every 10000 steps
#    model.save(f"{models_dir}/{TIMESTEPS*i}")

model_path = f"{models_dir}/170000.zip"

pi = PPO.load(model_path, env=env)

for ep in range(1,3000):
    obs = env.reset()
    done = False
    #while not done:
    env.mj_render()
    obs = env.get_obs()
    #action = model.get_action(obs)[0]
    action, _ = pi.predict(obs)
    next_o, r, done, *_, ifo = env.step(action)

env.close()
