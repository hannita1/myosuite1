from myosuite.utils import gym; register=gym.register
from my_hand_env import MyHandEnv
import os
curr_dir = os.path.dirname(os.path.abspath(__file__))
import numpy as np


# registering the base environment
register(
    id='MyHandEnv-v0',
    entry_point='my_hand_env:MyHandEnv',
    max_episode_steps=100,
    kwargs={
        'model_path': curr_dir + '/../simhive/myo_sim/hand/myohand.xml',
    }
)



# register variant environment with adapted parameters
from env_variants import register_env_variant

base_env_name = "MyHandEnv-v0"

base_env_variants = {
    'max_episode_steps': 200,  # Episode length
}

variant_env_name = register_env_variant(env_id=base_env_name, variants=base_env_variants)


# testing registered environment
env = gym.make("MyHandEnv-v0")
env.reset()
for _ in range(10000):
    env.mj_render()
    action = env.action_space.sample()
    next_o, r, done, *_, info = env.step(action)
    if done:
        break
env.close()