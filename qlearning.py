import os
#os.environ['QT_QPA_PLATFORM'] = 'wayland'
#os.environ['QT_QPA_PLATFORM'] = 'offscreen'
import gymnasium
import aisd_examples      
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time

env = gymnasium.make("aisd_examples/CreateRedBall-v0")

num_states = env.observation_space.n
num_actions = env.action_space.n

# Q-table initialization
q_table = np.zeros((num_states, num_actions))

# Original Hyperparameters
#episodes = 30     # Number of episodes
#alpha = 0.6        # Learning rate
#gamma = 0.9        # Discount factor
#epsilon = 0.1      # Exploration rate
#epsilon_plot = 0.1  
#decay = 0.1

episodes = 100     # Number of episodes
alpha = 0.7        # Learning rate
gamma = 0.95        # Discount factor
epsilon = 1.0      # Exploration rate
epsilon_plot = 1.0  
decay = 0.02

# Custom Hyperparameters 1
# episodes = 100   
# alpha = 0.7      
# gamma = 0.95      
# epsilon = 0.3   
# epsilon_plot = 0.3
# decay = 0.05     

# Custom Hyperparameters 2
# episodes = 75   
# alpha = 0.3      
# gamma = 0.9     
# epsilon_plot = 0.1
# epsilon = 0.1    
# decay = 0.15     

# Custom Hyperparameters 3
# episodes = 100 
# alpha = 0.5     
# gamma = 0.95     
# epsilon = 0.5    
# decay = 0.2      
# epsilon_plot = 0.5

episode_returns = []
episode_steps = []

for ep in range(episodes):
    state, info = env.reset()
    done = False
    total_reward = 0
    steps = 0

    while not done:
        os.system('clear')
        print(f"Episode #{ep+1}/{episodes}")
    

        steps += 1

        # Epsilon-greedy action selection
        if np.random.rand() < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(q_table[state])

        next_state, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward

        # Q-learning update
        best_next_q = np.max(q_table[next_state])
        q_table[state, action] = (1 - alpha) * q_table[state, action] + \
                                 alpha * (reward + gamma * best_next_q)

        state = next_state
        done = terminated or truncated

    # Decay epsilon
    epsilon -= decay * epsilon

    episode_returns.append(total_reward)
    episode_steps.append(steps)

    print(f"Episode {ep+1}/{episodes}, Steps: {steps}, Return: {total_reward}")
    time.sleep(0.3)

env.close()

# Define the hyperparameter string
hyperparams = f"HYPERPARAMETERS\n\nEpisodes={episodes}, \nα(Learning Rate)={alpha}, \nγ(Discount Factor)={gamma}, \nε(Exploration Rate)={epsilon_plot}, \nDecay={decay}"

# Plot results
plt.figure(figsize=(15,6))
plt.subplot(1,2,1)
plt.plot(episode_returns, label='Returns')
plt.xlabel('Episode')
plt.ylabel('Total Return')
plt.title(f'Returns')
plt.legend()

plt.subplot(1,2,2)
plt.plot(episode_steps, label='Steps', color='orange')
plt.xlabel('Episode')
plt.ylabel('Steps')
plt.title(f'Steps per Episode')
plt.legend()

plt.figtext(0.85, 0.5, hyperparams, fontsize=10, ha="left", bbox={"facecolor": "white", "alpha": 0.5, "pad": 5})

# General title for the whole figure
plt.suptitle("Q-Learning Performance (Original Hyperparameters)", fontsize=14, fontweight='bold')
plt.tight_layout(rect=[0, 0, 0.8, 0.95])
#plt.show()
plt.savefig("qlearning_performance.png")
