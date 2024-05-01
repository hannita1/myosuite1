from stable_baselines3 import PPO
import myosuite
import gym
import numpy as np

#pth = 'myosuite/agents/baslines_NPG/'
#policy = pth+"myoElbowPose1D6MRandom-v0/2022-02-26_21-16-27/35_env=myoElbowPose1D6MRandom-v0,seed=3/iterations/best_policy.pickle"
#policy = pth+"myoElbowPose1D6MExoRandom-v0/2022-02-26_21-16-27/36_env=myoElbowPose1D6MExoRandom-v0,seed=1/iterations/best_policy.pickle"
#pi = pickle.load(open(policy, 'rb'))



env = gym.make('myoElbowPose1D6MExoRandom-v0')

policy = "ElbowPose_policy.zip"
pi = PPO.load(policy)

episodes = 100
env.reset()
for episode in range(1, episodes+1):
    state = env.reset()
    done = False
    score = 0

    #while not done:
    env.mj_render()

    o = env.get_obs()
    action = pi.predict(o)[0]
    obs, reward, done, info = env.step(action)
    score += reward
    print('Episode: {}, Score: {}'.format(episode, score))

env.close()


