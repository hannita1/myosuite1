from myosuite.utils import gym; register=gym.register
from env_base import MujocoEnv
from myo.base_v0 import BaseV0
import numpy as np
# base environment


class MyHandEnv(BaseV0):
    def __init__(self, model_path='/Users/hannah/PycharmProjects/myosuite1/myosuite/simhive/myo_sim/hand/myohand.xml', obsd_model_path=None, seed=None):
        super(MyHandEnv, self).__init__(model_path, obsd_model_path, seed)
        self._setup(
            obs_keys=['joint_pos', 'joint_vel'],
            weighted_reward_keys={'reward_key': 1.0},
            frame_skip=5,
        )

    def get_obs_dict(self, sim):
        obs_dict = {}
        obs_dict['time'] = np.array([self.sim.data.time])
        obs_dict['joint_pos'] = sim.data.qpos.ravel().copy()
        obs_dict['joint_vel'] = sim.data.qvel.ravel().copy()
        obs_dict['act'] = sim.data.act.ravel().copy()
        print(f"get_obs_dict: {obs_dict}")
        return obs_dict

    def get_reward_dict(self, obs_dict):
        rwd_dict = {}
        rwd_dict['reward_key'] = -np.sum(np.square(obs_dict['joint_pos']))  # Beispielhafte Belohnungsfunktion
        rwd_dict['dense'] = rwd_dict['reward_key']
        rwd_dict['sparse'] = rwd_dict['reward_key'] > -1.0
        rwd_dict['solved'] = rwd_dict['reward_key'] > -0.1
        rwd_dict['done'] = rwd_dict['reward_key'] > -0.1

        rwd_dict['done'] = False

        print(f"Reward dict: {rwd_dict}")

        return rwd_dict



    def reset(self):
        self.sim.reset()


        # setting the initial state
        qpos = self.init_qpos.copy()
        qvel = self.init_qvel.copy()
        self.sim.data.qpos[:] = qpos
        self.sim.data.qvel[:] = qvel

        self.sim.forward()

        self.sim_obsd.data.qpos[:] = qpos
        self.sim_obsd.data.qvel[:] = qvel
        self.sim_obsd.forward()
        obs = self.get_obs()

        print(f"Reset: qpos={self.init_qpos}, qvel={self.init_qvel}, obs={obs}")
        return obs