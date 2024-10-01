
from myosuite.utils import gym
env = gym.make('MyoHandWineglassLift-v0')
env.reset()

for _ in range(1000):
  env.mj_render()
  env.step(env.action_space.sample()) # take a random action
env.close()
'''

from myosuite import myosuite_myodm_suite
for env in myosuite_myodm_suite:
    print(env)
    
'''