import numpy as np
import os
import myosuite
from myosuite.utils import gym
from stable_baselines3 import PPO
import time
import numpy as np
import matplotlib.pyplot as plt

from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3 import PPO, A2C

from stable_baselines3.common.evaluation import evaluate_policy


def train_and_evaluate():
    models_dir = "models/PPO/HandPoseFixed"
    logs_dir = "logs"

    if not os.path.exists(models_dir):
        os.makedirs(models_dir)

    env_id = 'myoHandPoseFixed-v0'

    # Number of parallel environments
    n_procs = 4

    # Use SubprocVecEnv for parallel environments
    # SubprocVecEnv creates several instances of the same environment & executes them parallely
    train_env = SubprocVecEnv([lambda: gym.make(env_id) for _ in range(n_procs)], start_method='fork')


    # Initialize the PPO model
    model = PPO('MlpPolicy', train_env, verbose=1, tensorboard_log=logs_dir)

    TIMESTEPS = 10
    # more timesteps -> more time to train per iteration, model has more time to learn before saving
    # -> more opportunities to learn from experience to improve the policy
    n_iterations = 3

    # List to store mean rewards
    mean_rewards = []
    std_rewards = []

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

    train_env.close()

    return model, models_dir,TIMESTEPS, n_iterations, env_id, mean_rewards, std_rewards

def plot_mean_rewards(mean_rewards, std_rewards):
    plt.figure(figsize=(10, 5))
    plt.errorbar(range(1, len(mean_rewards) + 1), mean_rewards, yerr=std_rewards, fmt='-o', capsize=5)
    plt.xlabel('Iteration')
    plt.ylabel('Mean Reward')
    plt.title('Policy Improvement Over Time')
    plt.grid(True)
    plt.savefig("policy_improvement.png")  # Speichert den Plot in einer Datei anstatt ihn anzuzeigen
    plt.close()




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
    #model_path = f"{models_dir}/90000"
    pi = PPO.load(model_path, env=render_env)

    for ep in range(1,3000):
        obs = render_env.reset()
        done = False
        #while not done:
        render_env.mj_render()
        obs = render_env.get_obs()
        #action = model.get_action(obs)[0]
        action, _ = pi.predict(obs)
        next_o, r, done, *_, ifo = render_env.step(action)


    render_env.close()

    # Evaluate the trained model
    mean_reward, std_reward = evaluate_policy(pi, render_env, n_eval_episodes=10)
    print(f"Mean reward: {mean_reward} +/- {std_reward:.2f}")

if __name__ == "__main__":
    #render_and_evaluate()
    #model, models_dir, TIMESTEPS, n_iterations, env_id, mean_rewards, std_rewards = train_and_evaluate()
    #plot_mean_rewards(mean_rewards, std_rewards)
    #model_path = f"{models_dir}/{TIMESTEPS * n_iterations}"
    models_dir = "models/PPO/HandPoseFixed"
    env_id = 'myoHandPoseFixed-v0'

    model_path = f"{models_dir}/90000"
    render_and_evaluate(model_path, env_id)

