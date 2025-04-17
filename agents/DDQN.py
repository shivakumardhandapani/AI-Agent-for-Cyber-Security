
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import os
import pandas as pd
from collections import deque
import matplotlib.pyplot as plt
import datetime


class DDQNConfig:
    def __init__(self):
        self.gamma = 0.99
        self.lr = 0.001
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.batch_size = 64
        self.memory_size = 10000
        self.target_update_freq = 1000
        self.hidden_dim = 128

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)

class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class DDQNAgent:
    def __init__(self, env, config):
        self.env = env
        self.config = config
        obs = env.reset()[0]
        self.state_dim = np.array(obs).flatten().shape[0]
        self.action_dim = env.action_space.n
        self.q_net = QNetwork(self.state_dim, self.action_dim, config.hidden_dim)
        self.target_net = QNetwork(self.state_dim, self.action_dim, config.hidden_dim)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=config.lr)
        self.memory = ReplayBuffer(config.memory_size)
        self.epsilon = config.epsilon
        self.hack_probs = []
        self.episode_lengths = []

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        state = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = self.q_net(state)
        return q_values.max(1)[1].item()

    def update(self):
        if len(self.memory) < self.config.batch_size:
            return
        batch = self.memory.sample(self.config.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions).unsqueeze(1)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)

        curr_q = self.q_net(states).gather(1, actions).squeeze()
        next_q = self.target_net(next_states).max(1)[0]
        target_q = rewards + (1 - dones) * self.config.gamma * next_q

        loss = F.smooth_l1_loss(curr_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


    def train_loop_simple(self, num_episodes=10000, max_steps=500, log_frequency=100):
        rewards, defender_rewards, epsilons = [], [], []
        hack_probs = []
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.output_dir = os.path.join("ddqn_results", f"run_{timestamp}")
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "plots"), exist_ok=True)

    
        for ep in range(num_episodes):
            obs = self.env.reset()[0]
            state = np.array(obs).flatten()
            total_reward, def_reward, steps = 0, 0, 0
    
            for _ in range(max_steps):
                action = self.select_action(state)
                def_action = self.env.defender_action_space.sample()
                next_obs, reward, done, _, info = self.env.step((action, def_action))
                next_state = np.array(next_obs).flatten()
    
                # Inside the loop:
                attacker_rew = reward[0] if isinstance(reward, (tuple, list, np.ndarray)) else reward
                defender_rew = reward[1] if isinstance(reward, (tuple, list, np.ndarray)) and len(reward) > 1 else 0
                
    
                # attacker_rew = reward[0] if isinstance(reward, (tuple, list, np.ndarray)) else reward
                # defender_rew = reward[1] if isinstance(reward, (tuple, list, np.ndarray)) and len(reward) > 1 else 0
    
                self.memory.add(state, action, attacker_rew, next_state, done)

                self.update()
    
                total_reward += attacker_rew
                def_reward += defender_rew
                state = next_state
                steps += 1
    
                if done:
                    break
    
            rewards.append(total_reward)
            defender_rewards.append(def_reward)
            epsilons.append(self.epsilon)
    
            if hasattr(info, "get") and "attacker_success" in info:
                hack_probs.append(1 if info["attacker_success"] else 0)
            else:
                hack_probs.append(0)
            self.epsilon = max(self.config.epsilon_min, self.epsilon * self.config.epsilon_decay)

    
            # self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
            if (ep + 1) % log_frequency == 0:
                print(f"Episode {ep+1}: AvgAttackerReward = {np.mean(rewards[-log_frequency:]):.2f}, "
                      f"AvgDefReward = {np.mean(defender_rewards[-log_frequency:]):.2f}, "
                      f"Epsilon = {self.epsilon:.3f}")

               
        results_df = pd.DataFrame({
            "episode": np.arange(1, num_episodes + 1),
            "attacker_reward": rewards,
            "defender_reward": defender_rewards,
            "epsilon": epsilons,
            "hack_probability": hack_probs
        })

        csv_path = os.path.join(self.output_dir, "training_results.csv")
        results_df.to_csv(csv_path, index=False)
        print(f"✅ Results saved to {csv_path}")

        self.plot_ddqn_training(results_df)
        return results_df
        # return df


    def update_network(self):
        if len(self.buffer) < self.config.batch_size:
            return

        states, actions, rewards, next_states, dones = self.buffer.sample(self.config.batch_size)

        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        curr_q = self.q_net(states).gather(1, actions).squeeze()
        next_q = self.target_net(next_states).max(1)[0]
        target_q = rewards + (1 - dones) * self.config.gamma * next_q

        loss = F.smooth_l1_loss(curr_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.train_steps += 1
        if self.train_steps % self.config.target_update_freq == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

    def plot_ddqn_training(self, df):
        if not os.path.exists(self.output_dir):
            print(f"❌ Output folder '{self.output_dir}' not found.")
            return

        plots_dir = os.path.join(self.output_dir, "plots")
        os.makedirs(plots_dir, exist_ok=True)

    
        plt.figure(figsize=(10, 4))
        plt.plot(df["attacker_reward"], label="Attacker Reward")
        plt.title("Attacker Reward per Episode")
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.grid(True)
        plt.tight_layout()
        plt.legend()
        attacker_path = os.path.join(plots_dir, "attacker_reward.png")
        plt.savefig(attacker_path)
        plt.show()

        # Plot Defender Reward
        if "defender_reward" in df.columns:
            plt.figure(figsize=(10, 4))
            plt.plot(df["defender_reward"], label="Defender Reward", color="orange")
            plt.title("Defender Reward per Episode")
            plt.xlabel("Episode")
            plt.ylabel("Reward")
            plt.grid(True)
            plt.tight_layout()
            plt.legend()
            defender_path = os.path.join(plots_dir, "defender_reward.png")
            plt.savefig(defender_path)
            plt.show()

        # Plot Epsilon Decay
        plt.figure(figsize=(10, 4))
        plt.plot(df["epsilon"], label="Epsilon", color="green")
        plt.title("Epsilon Decay")
        plt.xlabel("Episode")
        plt.ylabel("Epsilon")
        plt.grid(True)
        plt.tight_layout()
        epsilon_path = os.path.join(plots_dir, "epsilon_decay.png")
        plt.savefig(epsilon_path)
        plt.show()

        # Plot Hack Probability
        if "hack_probability" in df.columns:
            plt.figure(figsize=(10, 4))
            plt.plot(df["hack_probability"], label="Hack Probability", color="red")
            plt.title("Hack Probability Over Episodes")
            plt.xlabel("Episode")
            plt.ylabel("Hack Probability")
            plt.grid(True)
            plt.tight_layout()
            plt.legend()
            hack_path = os.path.join(plots_dir, "hack_probability.png")
            plt.savefig(hack_path)
            plt.show()


    def save_gif(env, agent, filename="agent_run.gif", num_steps=500):
        import imageio
        frames = []
        obs = env.reset()[0]
        state = np.array(obs).flatten()
    
        for _ in range(num_steps):
            frame = env.render(mode="rgb_array")
            frames.append(frame)
            action = agent.select_action(state)
            def_action = env.defender_action_space.sample()
            obs, _, done, _, _ = env.step((action, def_action))
            state = np.array(obs).flatten()
            if done:
                break
    
        imageio.mimsave(filename, frames, fps=30)



    
    
    
    
    
    
    
    
    
    
    


















# import numpy as np
# import torch
# import torch.nn as nn
# import torch.optim as optim
# import torch.nn.functional as F
# import os
# import random
# import time
# import matplotlib.pyplot as plt
# import pandas as pd

# class DDQNConfig:
#     def __init__(self, gamma=0.99, lr=0.0001, epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995,
#                  batch_size=64, buffer_capacity=10000, target_update_freq=1000, hidden_dim=128):
#         self.gamma = gamma
#         self.lr = lr
#         self.epsilon = epsilon_start
#         self.epsilon_end = epsilon_end
#         self.epsilon_decay = epsilon_decay
#         self.batch_size = batch_size
#         self.buffer_capacity = buffer_capacity
#         self.target_update_freq = target_update_freq
#         self.hidden_dim = hidden_dim

# class ReplayBuffer:
#     def __init__(self, capacity):
#         self.capacity = capacity
#         self.buffer = []
#         self.pos = 0

#     def push(self, state, action, reward, next_state, done):
#         if len(self.buffer) < self.capacity:
#             self.buffer.append(None)
#         self.buffer[self.pos] = (state, action, reward, next_state, done)
#         self.pos = (self.pos + 1) % self.capacity

#     def sample(self, batch_size):
#         batch = random.sample(self.buffer, batch_size)
#         return zip(*batch)

#     def __len__(self):
#         return len(self.buffer)

# class QNetwork(nn.Module):
#     def __init__(self, state_dim, action_dim, hidden_dim):
#         super().__init__()
#         self.fc1 = nn.Linear(state_dim, hidden_dim)
#         self.fc2 = nn.Linear(hidden_dim, hidden_dim)
#         self.fc3 = nn.Linear(hidden_dim, action_dim)

#     def forward(self, x):
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         return self.fc3(x)

# class DDQNAgent:
#     def __init__(self, env, config):
#         self.env = env
#         self.config = config
#         obs = self.env.reset()[0] if isinstance(env.reset(), tuple) else env.reset()
#         self.state_dim = np.array(obs).flatten().shape[0]
#         self.action_dim = self.env.action_space.n
#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#         self.q_net = QNetwork(self.state_dim, self.action_dim, config.hidden_dim).to(self.device)
#         self.target_net = QNetwork(self.state_dim, self.action_dim, config.hidden_dim).to(self.device)
#         self.target_net.load_state_dict(self.q_net.state_dict())
#         self.target_net.eval()

#         self.optimizer = optim.Adam(self.q_net.parameters(), lr=config.lr)
#         self.buffer = ReplayBuffer(config.buffer_capacity)
#         self.epsilon = config.epsilon
#         self.train_steps = 0

#         print(f"State dim: {self.state_dim}, Action dim: {self.action_dim}")

#     def select_action(self, state):
#         if random.random() < self.epsilon:
#             return random.randint(0, self.action_dim - 1)
#         state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
#         with torch.no_grad():
#             q_values = self.q_net(state)
#         return q_values.argmax().item()

#     def train_loop_simple(self, num_episodes=1000, max_steps=500, log_frequency=100):
#         rewards, epsilons, hack_probs = [], [], []
#         csv_log = []

#         for ep in range(1, num_episodes + 1):
#             obs = self.env.reset()[0] if isinstance(self.env.reset(), tuple) else self.env.reset()
#             state = np.array(obs).flatten()
#             total_reward = 0
#             steps = 0
#             hack_success = 0

#             for t in range(max_steps):
#                 action = self.select_action(state)
#                 def_action = self.env.defender_action_space.sample() if hasattr(self.env, "defender_action_space") else 0
#                 next_obs, reward, terminated, truncated, info = self.env.step((action, def_action))
#                 done = terminated or truncated
#                 next_state = np.array(next_obs).flatten()

#                 if isinstance(reward, (list, tuple, np.ndarray)):
#                     reward = reward[0]

#                 self.buffer.push(state, action, reward, next_state, done)
#                 self.update_network()
#                 state = next_state
#                 total_reward += reward
#                 steps += 1

#                 if "attacker_success" in info and info["attacker_success"]:
#                     hack_success += 1

#                 if done:
#                     break

#             self.epsilon = max(self.config.epsilon_end, self.epsilon * self.config.epsilon_decay)

#             rewards.append(total_reward)
#             epsilons.append(self.epsilon)
#             hack_probs.append(hack_success / steps if steps > 0 else 0)

#             if ep % log_frequency == 0:
#                 print(f"Episode {ep}: AvgReward = {np.mean(rewards[-log_frequency:]):.2f}, Epsilon = {self.epsilon:.3f}")

#         # Save to CSV
#         results_df = pd.DataFrame({
#             "episode": np.arange(1, num_episodes + 1),
#             "total_reward": rewards,
#             "epsilon": epsilons,
#             "hack_probability": hack_probs
#         })
#         csv_path = "training_results.csv"
#         results_df.to_csv(csv_path, index=False)
#         print(f"✅ Results saved to {csv_path}")

#         # Plot
#         self._plot_training(results_df)
#         return results_df

#     def update_network(self):
#         if len(self.buffer) < self.config.batch_size:
#             return

#         states, actions, rewards, next_states, dones = self.buffer.sample(self.config.batch_size)

#         states = torch.FloatTensor(np.array(states)).to(self.device)
#         actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
#         rewards = torch.FloatTensor(rewards).to(self.device)
#         next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
#         dones = torch.FloatTensor(dones).to(self.device)

#         curr_q = self.q_net(states).gather(1, actions).squeeze()
#         next_q = self.target_net(next_states).max(1)[0]
#         target_q = rewards + (1 - dones) * self.config.gamma * next_q

#         loss = F.smooth_l1_loss(curr_q, target_q)

#         self.optimizer.zero_grad()
#         loss.backward()
#         self.optimizer.step()

#         self.train_steps += 1
#         if self.train_steps % self.config.target_update_freq == 0:
#             self.target_net.load_state_dict(self.q_net.state_dict())

#     def _plot_training(self, df):
#         plt.figure(figsize=(12, 4))
#         plt.subplot(1, 3, 1)
#         plt.plot(df["total_reward"])
#         plt.title("Total Reward per Episode")
#         plt.xlabel("Episode")
#         plt.ylabel("Reward")
#         plt.savefig(os.path.join(output_dir, "reward_plot.png"))

#         plt.subplot(1, 3, 2)
#         plt.plot(df["epsilon"])
#         plt.title("Epsilon Decay")
#         plt.xlabel("Episode")
#         plt.ylabel("Epsilon")
#         plt.savefig(os.path.join(output_dir, "epsilon_decay.png"))

#         plt.subplot(1, 3, 3)
#         plt.plot(df["hack_probability"])
#         plt.title("Hack Probability")
#         plt.xlabel("Episode")
#         plt.ylabel("Probability")
#         plt.savefig(os.path.join(output_dir, "hack_probability.png"))

#         plt.tight_layout()
#         plt.show()
