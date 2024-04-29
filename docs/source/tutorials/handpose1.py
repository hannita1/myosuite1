import myosuite
import gym
import numpy as np
import os


import mujoco
env = gym.make('myoHandPoseFixed-v0', normalize_act = False)

env.env.init_qpos[:] = np.zeros(len(env.env.init_qpos),)
#env.env.init_qpos[:] = [9, 9, 9, -9.0904, 0.0824475, -0.681555, -0.514888, 0, -0.013964, -0.0458132, 0, 0.67553, -0.020944, 0.76979, 0.65982, 0, 0, 0, 0, 0.479155, -0.099484, 0.95831, 0]

print(env.env.init_qpos) #string of 23 zeros -> the 23 joints?
mjcModel = env.env.sim.model


print("Muscles:")
for i in range(mjcModel.na):
    print([i,mjcModel.actuator(i).name])


print("\nJoints:")
for i in range(mjcModel.njnt):
    print([i,mjcModel.joint(i).name])

#to hold phone -> Flexor Digitorum Profundus
    #FDP5 -> little finger
    #FDP4 -> ring finger
    #FDP3 -> middle finger
    #FDP2 -> index finger
#thumb? EPD, EPL, FPL
    # EPB -> Extensor Pollicis Brevis
    # EPL -> Extensor Pollicis Longus
    # FPL -> Flexor Pollicis Longus
    # APL -> Abductor Pollicis Longus

#EDC2 -> "handgelenk" + finger ->2,3,4,5

musc_fe = [mjcModel.actuator('FDP2').id ,mjcModel.actuator('FDP3').id ,mjcModel.actuator('FDP4').id ,mjcModel.actuator('FDP5').id ,mjcModel.actuator('EPB').id]
L_range = round(1/mjcModel.opt.timestep)
#skip_frame = 50
env.reset()

#frames_sim = []
for iter_n in range(30):
    print("iteration: " + str(iter_n))
    res_sim = []
    for rp in range(2):  # alternate between flexor and extensor
        for s in range(L_range):
            ctrl = np.zeros(mjcModel.na, )
            act_val = 0.15  # maximum muscle activation
            if rp == 0:
                ctrl[musc_fe[0]] = act_val
                ctrl[musc_fe[1]] = act_val
                ctrl[musc_fe[2]] = act_val
                ctrl[musc_fe[3]] = act_val
                ctrl[musc_fe[4]] = 0
            else:
                ctrl[musc_fe[4]] = act_val
                ctrl[musc_fe[0]] = 0
                ctrl[musc_fe[1]] = 0
                ctrl[musc_fe[2]] = 0
                ctrl[musc_fe[3]] = 0
            env.mj_render()

            #step() defined in myosuite/robot/robot.py
            env.step(ctrl)

            #print("ctrl: " + str(ctrl))



# Create the environment
env = gym.make('myoHandPoseFixed-v0')

# Inspect the observation space and action space
#print("Observation Space:", env.observation_space) #Box(108,)
#print("Action Space:", env.action_space) #Box(39,)


# Modify the initial pose directly
#env.env.init_qpos[:] = [0, 0, 0, -0.0904, 0.0824475, -0.681555, -0.514888, 0, -0.013964, -0.0458132, 0, 0.67553, -0.020944, 0.76979, 0.65982, 0, 0, 0, 0, 0.479155, -0.099484, 0.95831, 0]

# Reset the environment
obs = env.reset()

# Modify the initial pose directly
#env.env.init_qpos[:] = [0, 0, 0, -0.0904, 0.0824475, -0.681555, -0.514888, 0, -0.013964, -0.0458132, 0, 0.67553, -0.020944, 0.76979, 0.65982, 0, 0, 0, 0, 0.479155, -0.099484, 0.95831, 0]
#env.env.init_qpos[:] = [9, 9, 9, -9.0904, 0.0824475, -0.681555, -0.514888, 0, -0.013964, -0.0458132, 0, 0.67553, -0.020944, 0.76979, 0.65982, 0, 0, 0, 0, 0.479155, -0.099484, 0.95831, 0]


# Get details about the environment setup
#print("Environment Setup:")
print("Reset Type:", env.reset_type)
#print("Target Type:", env.target_type)
#print("Pose Threshold:", env.pose_thd)
#print("Weight Body Name:", env.weight_bodyname)
#print("Weight Range:", env.weight_range)

# Reset the environment to obtain the initial state
initial_state = env.reset()
print("Initial State:", initial_state)


# Close the environment
env.close()
