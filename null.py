import gymnasium
import aisd_examples  # ensure this import registers the environment

# Create the environment via the Gymnasium registry.
env = gymnasium.make("aisd_examples/CreateRedBall-v0", render_mode="human")
observation, info = env.reset()

# Run the null agent, which takes random actions for 1000 steps.
for _ in range(1000):
    action = env.action_space.sample()  # sample a random action from the discrete action space
    observation, reward, terminated, truncated, info = env.step(action)
    
    # If the episode ends for any reason, reset the environment.
    if terminated or truncated:
        observation, info = env.reset()

env.close()
