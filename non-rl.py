import gymnasium
import aisd_examples  # ensure the environment is registered
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Create the environment via the Gymnasium registry.
env = gymnasium.make("aisd_examples/CreateRedBall-v0")

# Set the total number of timesteps for the null agent run.
total_steps_target = 10000

# Initialize lists to track returns and episode lengths.
episode_returns = []
episode_steps = []

total_steps = 0
episode_count = 0

# Run episodes until the total steps reach the target.
while total_steps < total_steps_target:
    observation, info = env.reset()
    episode_return = 0
    episode_step_count = 0
    terminated = False
    truncated = False
    
    # Run one episode until termination, truncation, or total_steps limit.
    while not (terminated or truncated) and total_steps < total_steps_target:
        action = env.action_space.sample()  # select a random action
        observation, reward, terminated, truncated, info = env.step(action)
        
        # Accumulate reward and count the steps.
        episode_return += reward
        episode_step_count += 1
        total_steps += 1
        
    # Record statistics for this episode.
    episode_returns.append(episode_return)
    episode_steps.append(episode_step_count)
    episode_count += 1
    print(f"Episode {episode_count} finished: Steps = {episode_step_count}, Return = {episode_return}")

env.close()

# --- Plotting the Results ---
plt.figure(figsize=(15, 6))

# Plot episode returns.
plt.subplot(1, 2, 1)
plt.plot(episode_returns, marker='o', label="Returns")
plt.xlabel("Episode")
plt.ylabel("Total Return")
plt.title("Episode Returns")
plt.legend()

# Plot episode lengths.
plt.subplot(1, 2, 2)
plt.plot(episode_steps, marker='o', color="orange", label="Steps")
plt.xlabel("Episode")
plt.ylabel("Steps per Episode")
plt.title("Episode Steps")
plt.legend()

# Create a hyperparameter summary string.
hyperparams = (
    f"HYPERPARAMETERS\n\n"
    f"Total Steps = {total_steps_target}\n"
    f"Agent = Random (Null Agent)"
)
plt.figtext(0.85, 0.5, hyperparams, fontsize=10, ha="left", 
            bbox={"facecolor": "white", "alpha": 0.5, "pad": 5})

plt.suptitle("Null Agent Performance on CreateRedBall-v0", fontsize=14, fontweight="bold")
plt.tight_layout(rect=[0, 0, 0.8, 0.95])
plt.savefig("null_agent_performance.png")

