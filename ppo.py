import os
import gymnasium as gym
import aisd_examples
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback, CallbackList
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# --- Custom Callback to Print Progress ---
class PrintProgressCallback(BaseCallback):
    """
    A custom callback that prints the current episode number and the total number of timesteps
    each time an episode ends.
    """
    def __init__(self, verbose=0):
        super(PrintProgressCallback, self).__init__(verbose)
        self.episode_count = 0

    def _on_step(self) -> bool:
        # The Monitor wrapper adds an "episode" key into each info dict when an episode ends.
        for info in self.locals.get("infos", []):
            if "episode" in info:
                self.episode_count += 1
                print(f"Episode {self.episode_count} finished after {self.num_timesteps} timesteps")
        return True

# --- Create and Wrap the Environment ---
monitor_filename = "ppo_monitor_redball.csv"
env = gym.make("aisd_examples/CreateRedBall-v0")
env = Monitor(env, filename=monitor_filename)  # Monitor automatically creates the file in the working directory

# Vectorize the environment (required for some SB3 algorithms)
vec_env = DummyVecEnv([lambda: env])

# Evaluation callback (optional)
eval_callback = EvalCallback(
    env, 
    best_model_save_path="./logs/",
    log_path="./logs/", 
    eval_freq=500,
    deterministic=True, 
    render=False
)

# Custom progress callback to print episodes and timesteps
progress_callback = PrintProgressCallback()

# Combine callbacks in a CallbackList
callback = CallbackList([eval_callback, progress_callback])

# --- Create the PPO Model ---
model = PPO(
    policy="MlpPolicy",
    env=env,
    verbose=1,
    n_steps=2048,             # Number of steps collected per update
    batch_size=64,            # Mini-batch size for optimization
    learning_rate=3e-4,       # Learning rate for gradient descent
    ent_coef=0.1,            # Entropy coefficient; higher encourages more exploration
    clip_range=0.2,
    gamma=0.96,
    tensorboard_log="./ppo_redball_tensorboard/"
)

# Train for a total of 10000 timesteps
model.learn(total_timesteps=10000, log_interval=4, callback=callback)
model.save("ppo_redball")
model = PPO.load("ppo_redball", env=vec_env)

# --- Test the Trained Model ---
obs, _ = env.reset()
for _ in range(100):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, _ = env.reset()

env.close()

# --- Plot the Logged Training Results ---
# Read the Monitor log file generated by the Monitor wrapper.
# Skip the first line which contains metadata.
data = pd.read_csv("ppo_monitor_redball.csv.monitor.csv", skiprows=1)

# Extract episode returns and steps.
episode_returns = data["r"].tolist()  # "r" holds the return of each episode
episode_steps = data["l"].tolist()      # "l" holds the episode length (number of timesteps)

# Define the hyperparameter summary to be included on the plot.
hyperparams = (
    f"HYPERPARAMETERS\n\n"
    f"Total Timesteps=10000,\n"
    f"Learning Rate=3e-4,\n"
    f"Gamma=0.96,\n"
    f"Entropy coefficient=0.1,\n"
    f"Batch Size=64\n"
    f"Number of Steps=2048"
)

# Create the plots.
plt.figure(figsize=(15, 6))

# Plot Returns per Episode.
plt.subplot(1, 2, 1)
plt.plot(episode_returns, label="Returns")
plt.xlabel("Episode")
plt.ylabel("Total Return")
plt.title("Returns")
plt.legend()

# Plot Steps per Episode.
plt.subplot(1, 2, 2)
plt.plot(episode_steps, label="Steps", color="orange")
plt.xlabel("Episode")
plt.ylabel("Steps")
plt.title("Steps per Episode")
plt.legend()

# Add hyperparameter text on the right side.
plt.figtext(0.85, 0.5, hyperparams, fontsize=10, ha="left", 
            bbox={"facecolor": "white", "alpha": 0.5, "pad": 5})

plt.suptitle("PPO Performance on CreateRedBall-v0", fontsize=14, fontweight="bold")
plt.tight_layout(rect=[0, 0, 0.8, 0.95])

# Save and display the figure.
plt.savefig("ppo_performance.png")

