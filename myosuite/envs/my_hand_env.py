import sys
import os
from myosuite.utils import gym; register=gym.register
from myosuite.envs.myo.base_v0 import BaseV0
import numpy as np
import collections




class MyHandEnv(BaseV0):
    def __init__(self, model_path='/Users/hannah/PycharmProjects/myosuite1/myosuite/simhive/myo_sim/hand/myohand.xml', obsd_model_path=None, seed=None):
        super(MyHandEnv, self).__init__(model_path, obsd_model_path, seed)
        self._setup(
            # which observation key to include in the observation space of the environment
            # -> position of the index finger tip
            obs_keys = ['qpos', 'qvel', 'pose_err', 'IFtip_pos'],
            weighted_reward_keys= {
                'bonus': 4.0,
                'act_reg': 1.0,
                'penalty': 50.0,
                'distance_to_touch_areas': 5.0
            },
            frame_skip=5,
        )

#    def get_obs_dict(self, sim):
#        obs_dict = {}
#        obs_dict['time'] = np.array([self.sim.data.time])
#        obs_dict['joint_pos'] = sim.data.qpos.ravel().copy()
#        obs_dict['joint_vel'] = sim.data.qvel.ravel().copy()
#        obs_dict['act'] = sim.data.act.ravel().copy()
#        print(f"get_obs_dict: {obs_dict}")
#        return obs_dict

    # calls BaseV0 setup method, initializes the environment with the given obs_keys, reward_keys and frame_skipping
    def _setup(self, obs_keys, weighted_reward_keys, frame_skip):
        super()._setup(obs_keys=obs_keys, weighted_reward_keys=weighted_reward_keys, frame_skip=frame_skip)
        # will store a list of site ids -> will store the ID for the site 'IFtip' & the targets
        self.tip_sids = [self.sim.model.site_name2id('IFtip')]
        self.target_sids = [self.sim.model.site_name2id(f'target_touch_area_{i + 1}') for i in range(3)]
        # usage:
        #   fingertip_pos = self.sim.data.site_xpos[self.tip_sids[0]]
        #   target_pos_1 = self.sim.data.site_xpos[self.target_sids[0]]

    def get_obs_dict(self, sim):

        obs_dict = {}
        obs_dict['time'] = np.array([sim.data.time])
        # joint positions
        obs_dict['qpos'] = sim.data.qpos[:].copy()
        # joint velocities
        obs_dict['qvel'] = sim.data.qvel[:].copy() * self.dt
        # actuator values
        obs_dict['act'] = sim.data.act[:].copy() if sim.model.na > 0 else np.zeros_like(obs_dict['qpos'])
        # sim.data.site_xpos -> an array that contains the global positions of all sites in the simulation
        # sim.model.site_name2id('IFtip') -> converts the name of a site into its numerical ID
        # -> copies the position of the IFtip into the obs_dict
        obs_dict['IFtip_pos'] = sim.data.site_xpos[sim.model.site_name2id('IFtip')].copy()


        # positions of the tocuh targets
        target_positions = [
            self.sim.data.site_xpos[self.sim.model.site_name2id('target_touch_area_1')],
            self.sim.data.site_xpos[self.sim.model.site_name2id('target_touch_area_2')],
            self.sim.data.site_xpos[self.sim.model.site_name2id('target_touch_area_3')],
        ]


        # calculate pose error (difference between current position of index finger tip and target positions)
        # since for now we only want to hit any target,
        # -> pose error is the distance between the target we are closest to and the index finger
        #obs_dict['pose_err'] = self.target_jnt_value - obs_dict['qpos']
        #obs_dict['pose_err'] = min(np.linalg.norm(obs_dict['IFtip_pos'] - target) for target in target_positions)
        dist_IF_to_touch_1 = np.linalg.norm(obs_dict['IFtip_pos'] - target_positions[0])
        dist_IF_to_touch_2 = np.linalg.norm(obs_dict['IFtip_pos'] - target_positions[1])
        dist_IF_to_touch_3 = np.linalg.norm(obs_dict['IFtip_pos'] - target_positions[2])
        min_dist_to_touch = min(dist_IF_to_touch_1, dist_IF_to_touch_2, dist_IF_to_touch_3)

        # Ensure pose_err is an array
        obs_dict['pose_err'] = np.array([min_dist_to_touch])
        #print(f"get_obs_dict: {obs_dict}")
        return obs_dict

    def get_reward_dict(self, obs_dict):
        # pose_err = euklidean distance between the current position of the finger tip and the target positions (touch areas)
        # pose_dist = norm of the pose_err
        pose_dist = float(np.linalg.norm(obs_dict['pose_err'], axis=-1))
        # act = vector describing all current activities of the joints
        act_mag = float(np.linalg.norm(obs_dict['act'], axis=-1))
        if self.sim.model.na != 0:
            # divided by the number of joints to get the average
            act_mag = act_mag / self.sim.model.na
        # threshold for when the distance between finger and target is too large
        far_th = 4 * np.pi / 2
        tap_threshold = 0.02

        rwd_dict = collections.OrderedDict((
            # distance (=error) between the position of the finger and the target, the closer to the target the smaller the error
            ('distance_to_touch_areas', -1.0 * pose_dist),
            # reward if the finger is within a certain distance to the target,
            ('bonus', 1.0 if pose_dist < tap_threshold else 0.0),
            # if the distance between the finger and the target is too big -> penalty
            ('penalty', -1.0 * (pose_dist > far_th)),
            # penalty for too much activity of the joints -> should prevent too much unnecessary movement
            # negative value of the norm of the joint activity,
            # the norm of the act vector is the sum of the activity of the joints
            ('act_reg', -1.0 * act_mag),
            # binary reward -> 1 if fingertip inside the threshold
            ('sparse', 1.0 if pose_dist < tap_threshold else 0.0),
            # true if task is solved -> the distance is lower than the threshold
            ('solved', pose_dist < tap_threshold),
            # episode is done when we are too far from target
            ('done', pose_dist > far_th),
        ))

        rwd_dict['dense'] = np.sum([wt * rwd_dict[key] for key, wt in self.rwd_keys_wt.items()])
        return rwd_dict







    """
    def reset(self, seed=None, options=None):
        #self.sim.reset()
        if seed is not None:
            self.seed(seed)


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
    """
