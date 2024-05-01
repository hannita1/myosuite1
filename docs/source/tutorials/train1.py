
import numpy as np
import os

import myosuite
import gym
from stable_baselines3 import PPO

#env = gym.make('myoElbowPose1D6MRandom-v0')
#env=gym.make('myoHandPoseFixed-v0')
env=gym.make('MyoHandPhoneFixed-v0')

env.reset()

# PPO = Proximal Policy Optimization
# -> RL algorithm used by myosuite tutorial, can be used for continous action spaces
# -> policy gradient algorithm
# MlpPolicy
# -> specifies policy network architecture used by the PPO algorithm
# -> "MlpPolicy" = Multi-Layer Perceptron Policy
# -> simple feedforward neural network, often used in RL algorithms
model = PPO("MlpPolicy", env, verbose=0)

print("========================================")
print("Starting policy learning")
print("========================================")

model.learn(total_timesteps=10000)

print("========================================")
print("Job Finished.")
print("========================================")

#model.save('HandPose_policy') #trainig finished -> trained model is saved in zip
#model.save('ElbowPose_policy')
model.save('MyoHandPhoneLift_policy')

#policy = "HandPose_policy.zip"
#policy = "ElbowPose_policy.zip"
policy = "MyoHandPhoneLift_policy.zip"

pi = PPO.load(policy) #model is loaded from zip

#are those the strating angles? because the target positions/angles are random right?
# but they are called target_jnt_value ???
AngleSequence = [60, 30, 30, 60, 80, 80, 60, 30, 80, 30, 80, 60]
#AngleSequence = [60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60]
#env.reset()

episodes = 100
score = 0
for ep in range(episodes):
#for ep in range(len(AngleSequence)):
    #print("Ep {} of {} testing angle {}".format(ep, len(AngleSequence), AngleSequence[ep]))

    # calculate the target joint angle -> iterates through angleSequence by episode
    # np.deg2rad converts angle for mujoco
    # but if I use the hand model I not only have a target angle but 5 target positions, 1 for each finger??
    # how can I define these target positions?
    #env.env.target_jnt_value = [np.deg2rad(AngleSequence[int(ep)])]

    ## To update target position
    # can only define target positions by joints?
    #env.env.target_jnt_value = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]
    #print("Target joint value: {}".format(env.env.target_jnt_value))
    #env.env.target_type = 'fixed'
    #env.env.weight_range=(0,0)
    #env.env.update_target()


    for _ in range(400):
        env.mj_render()

        obs = env.get_obs()
        #env.examine_policy(pi)
        action = pi.predict(obs)[0]
        next_obs, reward, done, info = env.step(action) # take an action based on the current observation
        score += reward
        #print('Score: {},Action: {}, Reward: {}\n'.format(score, action, reward))

env.close()