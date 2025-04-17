import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

# =============== REPLAY BUFFER ================
class ReplayBuffer:
    def __init__(self, capacity, fixed_len=None):
        self.buffer = deque(maxlen=capacity)
        self.fixed_len = fixed_len

    def push(self, state, action, reward, next_state, done):
        state = np.array(state).flatten()
        next_state = np.array(next_state).flatten()

        if self.fixed_len is None:
            self.fixed_len = len(state)

        state = self._fix_shape(state)
        next_state = self._fix_shape(next_state)

        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = map(np.array, zip(*batch))

        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)

        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)

    def _fix_shape(self, arr):
        if len(arr) < self.fixed_len:
            return np.pad(arr, (0, self.fixed_len - len(arr)))
        elif len(arr) > self.fixed_len:
            return arr[:self.fixed_len]
        return arr

# =============== Q-NETWORK ================
class DQNNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQNNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        return self.net(x)

# =============== DQN AGENT ================
class DQNAgent:
    def __init__(self, state_dim, action_dim, buffer_capacity=10000,
                 gamma=0.99, lr=1e-3, batch_size=64,
                 epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995, device="cpu"):

        self.device = device
        self.action_dim = action_dim
        self.batch_size = batch_size
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.expected_state_dim = state_dim


        self.q_net = DQNNetwork(state_dim, action_dim).to(self.device)
        self.target_net = DQNNetwork(state_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.buffer = ReplayBuffer(capacity=buffer_capacity)
        self.steps = 0

    def select_action(self, state):
        state = np.array(state).flatten()
        if random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_net(state_tensor)
        return q_values.argmax().item()

    def store(self, state, action, reward, next_state, done):
        self.buffer.push(state, action, reward, next_state, done)

    def update(self):
        if len(self.buffer) < self.batch_size:
            return

        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)

        states = states.to(self.device)
        actions = actions.unsqueeze(1).to(self.device)
        rewards = rewards.unsqueeze(1).to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.unsqueeze(1).to(self.device)

        q_values = self.q_net(states).gather(1, actions)
        with torch.no_grad():
            next_q = self.target_net(next_states).max(1, keepdim=True)[0]
            target = rewards + self.gamma * next_q * (1 - dones)

        loss = nn.MSELoss()(q_values, target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.steps += 1
        if self.steps % 100 == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def save(self, path="dqn_model.pt"):
        torch.save(self.q_net.state_dict(), path)
        print(f"Model saved to {path}")

    def load(self, path="dqn_model.pt"):
        self.q_net.load_state_dict(torch.load(path))
        self.target_net.load_state_dict(self.q_net.state_dict())
        print(f"Model loaded from {path}")

    # === Utility: Fix inconsistent obs shapes ===
    def fix_obs_shape(self, obs):
        obs = np.array(obs).flatten()
        if obs.shape[0] < self.expected_state_dim:
            padded = np.zeros(self.expected_state_dim)
            padded[:obs.shape[0]] = obs
            return padded
        elif obs.shape[0] > self.expected_state_dim:
            return obs[:self.expected_state_dim]
        return obs
