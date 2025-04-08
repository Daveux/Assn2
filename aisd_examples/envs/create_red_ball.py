import gymnasium as gym
from gymnasium import spaces

class CreateRedBallEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, render_mode=None):
        # Set up arbitrary discrete observation and action spaces.
        self.observation_space = spaces.Discrete(10)
        self.action_space = spaces.Discrete(5)
        self.render_mode = render_mode
        
        # Initialize the state.
        self.state = 0

    def reset(self, seed=None, options=None):
        # Call the parent reset to handle seeding.
        super().reset(seed=seed)
        
        # Choose an arbitrary state from the observation space.
        self.state = self.observation_space.sample()
        return self.state, {}

    def step(self, action):
        # Choose another arbitrary state.
        self.state = self.observation_space.sample()
        
        # Provide an arbitrary reward (here, a fixed reward of 1).
        reward = 1
        
        # Set termination and truncation arbitrarily.
        terminated = False
        truncated = False
        
        # Use an empty dict for info.
        info = {}
        print(f"Successfully connected to step \n Action taken: {action}, New state: {self.state}, Reward: {reward}")
        return self.state, reward, terminated, truncated, info

    def render(self):
        # Render does nothing for now.
        pass

    def close(self):
        # No external resources to close.
        pass
