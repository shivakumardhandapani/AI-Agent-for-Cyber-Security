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
        # Preserve original structure while converting to numpy array
        state = np.array(state, copy=False)
        next_state = np.array(next_state, copy=False)

        # Handle dynamic fixed length initialization
        if self.fixed_len is None:
            self.fixed_len = state.shape[0] if len(state.shape) > 0 else 1

        # Ensure consistent shapes
        state = self._fix_shape(state)
        next_state = self._fix_shape(next_state)

        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = map(np.array, zip(*batch))

        # Handle multi-dimensional states (e.g., images)
        states = torch.FloatTensor(np.stack(states))
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(np.stack(next_states))
        dones = torch.FloatTensor(dones)

        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)

    def _fix_shape(self, arr):
        # Handle both 1D and multi-dimensional arrays
        if isinstance(arr, np.ndarray) and arr.ndim > 1:
            current_len = arr.shape[0]
            if current_len < self.fixed_len:
                pad_width = [(0, self.fixed_len - current_len)] + [(0,0)]*(arr.ndim-1)
                return np.pad(arr, pad_width)
            elif current_len > self.fixed_len:
                return arr[:self.fixed_len]
            return arr
        else:
            # Handle 1D case
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
                 epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995, 
                 tau=0.005, device="cpu"):

        self.device = device
        self.action_dim = action_dim
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau  # For soft updates
        
        # Epsilon handling
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.expected_state_dim = state_dim

        # Networks
        self.q_net = DQNNetwork(state_dim, action_dim).to(self.device)
        self.target_net = DQNNetwork(state_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        # Optimization
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.loss_fn = nn.SmoothL1Loss()  # Huber loss
        self.buffer = ReplayBuffer(capacity=buffer_capacity)
        self.steps = 0

    def select_action(self, state):
        state = self.fix_obs_shape(state)
        if random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_net(state_tensor)
        return q_values.argmax().item()

    def update(self):
        if len(self.buffer) < self.batch_size:
            return

        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)
        
        # Convert to tensors
        states = states.to(self.device)
        actions = actions.unsqueeze(1).to(self.device)
        rewards = rewards.unsqueeze(1).to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.unsqueeze(1).to(self.device)

        # Q-value calculation
        q_values = self.q_net(states).gather(1, actions)
        
        # Target calculation with double DQN
        with torch.no_grad():
            next_actions = self.q_net(next_states).max(1)[1].unsqueeze(1)
            next_q = self.target_net(next_states).gather(1, next_actions)
            target = rewards + self.gamma * next_q * (1 - dones)

        # Loss calculation
        loss = self.loss_fn(q_values, target)

        # Gradient clipping
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), 10.0)
        self.optimizer.step()

        # Soft target network updates
        self.soft_update_target_network()

        # Adaptive epsilon decay
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        self.steps += 1

    def soft_update_target_network(self):
        """Soft update model parameters using tau"""
        for target_param, q_param in zip(self.target_net.parameters(), self.q_net.parameters()):
            target_param.data.copy_(self.tau * q_param.data + (1.0 - self.tau) * target_param.data)

    def store(self, state, action, reward, next_state, done):
        self.buffer.push(state, action, reward, next_state, done)

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
