
import os
import csv
import sys
import time
import numpy as np
import matplotlib.pyplot as plt

import myosuite
from myosuite.utils import gym
from myosuite.utils import gym; register=gym.register

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3 import PPO, A2C
from stable_baselines3.common.evaluation import evaluate_policy

#my_hand_env_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../../myosuite/envs'))
#sys.path.append(my_hand_env_path)
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if project_root not in sys.path:
    sys.path.append(project_root)
from myosuite.envs.my_hand_env import MyHandEnv

#model_path = os.path.join(project_root, 'myosuite/envs/myo/assets/hand/myohand_tabletop_phone.xml')

def train_and_evaluate():
    #models_dir = "models/PPO/ObjHoldFixed"
    #logs_dir = "logs/ObjHoldFixed"
    #results_file = "training_results_ObjHoldFixed.csv"
    models_dir = "models/PPO/MyHandEnv"
    logs_dir = "logs/MyHandEnv"
    results_file = "training_results_MyHandEnv.csv"

    if not os.path.exists(models_dir):
        os.makedirs(models_dir)

    #env_id = 'myoHandPoseFixed-v0'
    #env_id = 'myoHandObjHoldFixed-v0'
    env_id = 'MyHandEnv-v0'

    # Number of parallel environments
    n_procs = 10

    # Use SubprocVecEnv for parallel environments, class from stable_baselines3
    # SubprocVecEnv creates several instances of the same environment & executes them parallely
    train_env = SubprocVecEnv([lambda: gym.make(env_id) for _ in range(n_procs)], start_method='fork')
    # list of lambda functions, that create env_id n_procs times

    # Initialize the PPO model
    model = PPO('MlpPolicy', train_env, verbose=1, tensorboard_log=logs_dir)
    # MlpPolicy = multi Layer Perception Policy = a neural network
    # parallel environments train_env

    TIMESTEPS = 100
    # more timesteps -> more time to train per iteration, model has more time to learn before saving
    # -> more opportunities to learn from experience to improve the policy
    n_iterations = 10

    # List to store mean rewards
    mean_rewards = []
    std_rewards = []

    with open(results_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Iteration", "Mean Reward", "Std Reward"])

    for i in range(1, n_iterations):
        # more episode -> more models are saved, more models to evaluate
        model.learn(total_timesteps=TIMESTEPS)
        # Save model every 10000 steps
        model.save(f"{models_dir}/{TIMESTEPS * i}")

        # Evaluate the trained model
        mean_reward, std_reward = evaluate_policy(model, train_env, n_eval_episodes=10)
        mean_rewards.append(mean_reward)
        std_rewards.append(std_reward)
        print(f"Iteration {i}: Mean reward: {mean_reward} +/- {std_reward:.2f}")

        with open(results_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([i, mean_reward, std_reward])

    train_env.close()

    return model, models_dir,TIMESTEPS, n_iterations, env_id, mean_rewards, std_rewards

def plot_mean_rewards(mean_rewards, std_rewards):
    print(f"figure")
    plt.figure(figsize=(10, 5))
    print("error bar")
    #plt.errorbar(range(1, len(mean_rewards) + 1), mean_rewards, yerr=std_rewards, fmt='-o', capsize=5)
    plt.plot(range(1, len(mean_rewards) + 1), mean_rewards, '-o')
    print("iteration")
    plt.xlabel('Iteration')
    print("mean reward label")
    plt.ylabel('Mean Reward')
    print("title label")
    plt.title('Policy Improvement Over Time')
    print("grid")
    plt.grid(True)
    print("save")
    plt.savefig("policy_improvement.png")
    #plt.close()

def plot_mean_rewards_from_csv(results_file):
    iterations = []
    mean_rewards = []
    std_rewards = []

    with open(results_file, mode='r') as file:
        reader = csv.reader(file)
        next(reader)  # skip first row
        for row in reader:
            iterations.append(int(row[0]))
            mean_rewards.append(float(row[1]))
            std_rewards.append(float(row[2]))

    plt.figure(figsize=(10, 5))
    plt.errorbar(iterations, mean_rewards, yerr=std_rewards, fmt='-o', capsize=5, label='Mean Reward')
    plt.xlabel('Iteration')
    plt.ylabel('Mean Reward')
    plt.title('Policy Improvement Over Time')
    plt.grid(True)
    plt.legend()
    plt.savefig("policy_improvement.png")
    plt.show()


# Plot the mean rewards over time
#plot_mean_rewards(mean_rewards, std_rewards)

# Create a single environment for rendering
#render_env = gym.make(env_id)

# Load the trained model
#model_path = f"{models_dir}/90000"
#pi = PPO.load(model_path, env=render_env)

# Create a single environment for rendering
def render_and_evaluate(model_path, env_id):

    render_env = gym.make(env_id)

    # Load the last trained model
    pi = PPO.load(model_path, env=render_env)

    for ep in range(5000):
        print(f'Episode: {ep} of 5000')
        obs = render_env.reset()
        while True:
            render_env.mj_render()

            obs = render_env.get_obs()
            #action = render_env.action_space.sample()
            action, _ = pi.predict(obs)

            next_state, reward, done,*_, info = render_env.step(action)
            if done:
                break


    render_env.close()

    # Evaluate the trained model
    mean_reward, std_reward = evaluate_policy(pi, render_env, n_eval_episodes=10)
    print(f"Mean reward: {mean_reward} +/- {std_reward:.2f}")

if __name__ == "__main__":
    register(
        id='MyHandEnv-v0',
        entry_point='my_hand_env:MyHandEnv',
        max_episode_steps=100,
        kwargs={
            # 'model_path': curr_dir + '/../simhive/myo_sim/hand/myohand.xml',
            # 'model_path': curr_dir + '/myo/assets/hand/myohand_hold.xml',
            # 'model_path': curr_dir + '/myo/assets/hand/myohand_phone.xml',
            #'model_path': project_root + '/myo/assets/hand/myohand_tabletop_phone.xml',
            # 'model_path': curr_dir + '/myo/assets/hand/myohand_tabletop2.xml',
            'model_path': project_root + '/myosuite/envs/myo/assets/hand/myohand_tabletop_phone.xml'
        }
    )


    ## train model
    model, models_dir, TIMESTEPS, n_iterations, env_id, mean_rewards, std_rewards = train_and_evaluate()
    #print(f"mean reward: {mean_rewards} ")
    #plot_mean_rewards(mean_rewards, std_rewards)
    #model_path = f"{models_dir}/{TIMESTEPS * n_iterations}"



    ## Test trained policy
    models_dir = "models/PPO/MyHandEnv"
    env_id = 'MyHandEnv-v0'

    #models_dir = "models/PPO/HandPoseFixed"
    #env_id = 'myoHandPoseFixed-v0'

    #models_dir = "models/PPO/ObjHoldFixed"
    #env_id = 'myoHandObjHoldFixed-v0'

    ## load policy and render
    model_path = f"{models_dir}/900"
    render_and_evaluate(model_path, env_id)


    #plot_mean_rewards_from_csv("training_results.csv")

