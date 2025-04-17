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
        
        # Define a deeper network architecture
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
        )
        
        # Advantage stream (for each action)
        self.advantage_stream = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )
        
        # Value stream (state value)
        self.value_stream = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
    
    def forward(self, x):
        features = self.feature_extractor(x)
        
        advantage = self.advantage_stream(features)
        value = self.value_stream(features)
        
        # Combine value and advantage using the Dueling DQN architecture
        # Q(s,a) = V(s) + (A(s,a) - mean(A(s,a')))
        return value + advantage - advantage.mean(dim=1, keepdim=True)

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

        # Networks - using dueling architecture
        self.q_net = DQNNetwork(state_dim, action_dim).to(self.device)
        self.target_net = DQNNetwork(state_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        # Optimization
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.loss_fn = nn.SmoothL1Loss()  # Huber loss
        self.buffer = ReplayBuffer(capacity=buffer_capacity)
        
        # Training tracking
        self.steps = 0
        self.episodes_completed = 0
        self.cumulative_reward = 0
        self.episode_rewards = []
        self.training_losses = []
        
        # Reward normalization - initialized during training
        self.reward_mean = 0
        self.reward_std = 1
        self.reward_min = 0
        self.reward_max = 1
        self.normalize_rewards = True
        self.reward_normalization_steps = 1000  # Update normalization every N steps

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
            return None

        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)
        
        # Handle reward internally - no external manipulation needed
        if self.normalize_rewards and self.steps % self.reward_normalization_steps == 0:
            # Update reward statistics for normalization
            all_rewards = [item[2] for item in self.buffer.buffer]
            if len(all_rewards) > 0:
                self.reward_mean = np.mean(all_rewards)
                self.reward_std = np.std(all_rewards) if np.std(all_rewards) > 0 else 1.0
                self.reward_min = np.min(all_rewards)
                self.reward_max = np.max(all_rewards)
                
        # Convert tensors to device
        states = states.to(self.device)
        actions = actions.unsqueeze(1).to(self.device)
        rewards = rewards.unsqueeze(1).to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.unsqueeze(1).to(self.device)

        # Current Q-values
        q_values = self.q_net(states).gather(1, actions)
        
        # Double DQN: use online network to select actions, target network to evaluate
        with torch.no_grad():
            next_q_values = self.q_net(next_states)
            next_actions = next_q_values.max(1)[1].unsqueeze(1)
            next_q_targets = self.target_net(next_states).gather(1, next_actions)
            target_q_values = rewards + self.gamma * next_q_targets * (1 - dones)

        # Compute loss and update
        loss = self.loss_fn(q_values, target_q_values)
        loss_value = loss.item()
        self.training_losses.append(loss_value)
        
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), 10.0)
        self.optimizer.step()

        # Soft update target network
        self.soft_update_target_network()

        # Update epsilon (exploration rate)
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        self.steps += 1
        
        return loss_value

    def soft_update_target_network(self):
        """Soft update model parameters using tau"""
        for target_param, q_param in zip(self.target_net.parameters(), self.q_net.parameters()):
            target_param.data.copy_(self.tau * q_param.data + (1.0 - self.tau) * target_param.data)

    def store(self, state, action, reward, next_state, done):
        """Store experience in replay buffer with original rewards"""
        self.buffer.push(state, action, reward, next_state, done)
        
    def end_episode(self, episode_reward):
        """Track episode completion and rewards"""
        self.episodes_completed += 1
        self.cumulative_reward += episode_reward
        self.episode_rewards.append(episode_reward)
        
        # Adaptive exploration based on performance
        if len(self.episode_rewards) >= 10:
            recent_rewards = self.episode_rewards[-10:]
            if np.mean(recent_rewards) > 0 and self.episodes_completed > 100:
                # If doing well, reduce exploration more quickly
                self.epsilon = max(self.epsilon_end, self.epsilon * 0.95)

    def save(self, path="dqn_model.pt"):
        """Save model and training state"""
        torch.save({
            'q_net_state_dict': self.q_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'steps': self.steps,
            'epsilon': self.epsilon,
            'episodes_completed': self.episodes_completed,
            'reward_stats': {
                'mean': self.reward_mean,
                'std': self.reward_std,
                'min': self.reward_min,
                'max': self.reward_max
            },
            'training_losses': self.training_losses,
            'episode_rewards': self.episode_rewards
        }, path)
        print(f"Model saved to {path}")

    def load(self, path="dqn_model.pt"):
        """Load model and training state"""
        checkpoint = torch.load(path)
        self.q_net.load_state_dict(checkpoint['q_net_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.steps = checkpoint['steps']
        self.epsilon = checkpoint['epsilon']
        self.episodes_completed = checkpoint['episodes_completed']
        
        if 'reward_stats' in checkpoint:
            stats = checkpoint['reward_stats']
            self.reward_mean = stats['mean']
            self.reward_std = stats['std']
            self.reward_min = stats['min']
            self.reward_max = stats['max']
            
        if 'training_losses' in checkpoint:
            self.training_losses = checkpoint['training_losses']
            
        if 'episode_rewards' in checkpoint:
            self.episode_rewards = checkpoint['episode_rewards']
            
        print(f"Model loaded from {path}")

    # === Utility: Fix inconsistent obs shapes ===
    def fix_obs_shape(self, obs):
        """Ensure observation has consistent shape"""
        obs = np.array(obs).flatten()
        if obs.shape[0] < self.expected_state_dim:
            padded = np.zeros(self.expected_state_dim)
            padded[:obs.shape[0]] = obs
            return padded
        elif obs.shape[0] > self.expected_state_dim:
            return obs[:self.expected_state_dim]
        return obs
    
    def get_stats(self):
        """Get agent statistics"""
        return {
            'steps': self.steps,
            'episodes': self.episodes_completed,
            'epsilon': self.epsilon,
            'avg_reward': np.mean(self.episode_rewards[-10:]) if len(self.episode_rewards) >= 10 else 0,
            'reward_stats': {
                'mean': self.reward_mean,
                'std': self.reward_std,
                'min': self.reward_min,
                'max': self.reward_max
            }
        }