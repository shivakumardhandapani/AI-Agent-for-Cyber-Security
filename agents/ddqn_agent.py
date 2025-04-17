import numpy as np
import tensorflow as tf
import random
from collections import deque
import os
import datetime
import time
import logging
import gym


class ReplayBuffer:
    """
    Replay Buffer for storing transitions for training a DDQN agent
    """
    def __init__(self, max_size):
        """
        Initialize the replay buffer
        
        :param max_size: maximum size of the buffer
        """
        self.buffer = deque(maxlen=max_size)
    
    def push(self, state, action, reward, next_state, done):
        """
        Add a new transition to the buffer
        
        :param state: current state
        :param action: action taken
        :param reward: reward received
        :param next_state: next state
        :param done: if episode is done or not
        """
        experience = (state, action, reward, next_state, done)
        self.buffer.append(experience)
    
    def sample(self, batch_size):
        """
        Sample a batch of experiences from the buffer
        
        :param batch_size: size of the batch to sample
        :return: batch of experiences
        """
        batch = random.sample(self.buffer, min(len(self.buffer), batch_size))
        states, actions, rewards, next_states, dones = map(np.stack, zip(*batch))
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        """
        Get current size of the buffer
        
        :return: size of buffer
        """
        return len(self.buffer)


class DDQNNetwork(tf.keras.Model):
    """
    Neural Network model for the DDQN agent
    """
    def __init__(self, num_states, num_actions, hidden_layers=[64, 64], name="ddqn_network"):
        """
        Initialize the network
        
        :param num_states: dimension of state space
        :param num_actions: dimension of action space
        :param hidden_layers: list of hidden layer sizes
        :param name: name of the network
        """
        super(DDQNNetwork, self).__init__(name=name)
        
        # Define neural network layers
        self.hidden_layers = []
        for size in hidden_layers:
            self.hidden_layers.append(tf.keras.layers.Dense(size, activation='relu'))
        
        self.output_layer = tf.keras.layers.Dense(num_actions, activation=None)
    
    def call(self, inputs):
        """
        Forward pass through the network
        
        :param inputs: input to the network (state)
        :return: Q-values for each action
        """
        x = inputs
        for layer in self.hidden_layers:
            x = layer(x)
        
        q_values = self.output_layer(x)
        return q_values


class DDQNConfig:
    """
    Configuration for the DDQN agent
    """
    def __init__(self, gamma=0.99, lr=0.00001, batch_size=32, epsilon=1.0, epsilon_decay=0.999,
                 min_epsilon=0.01, target_network_update_freq=100, replay_memory_size=10000,
                 num_episodes=10000, eval_frequency=1000, eval_episodes=100, train_log_frequency=100,
                 eval_log_frequency=1, eval_render=False, eval_sleep=0.0, render=False,
                 attacker=True, defender=False, save_dir="./results/data", save_frequency=1000,
                 hidden_layers=[64, 64]):
        """
        Initialize the configuration

        :param gamma: discount factor
        :param lr: learning rate for the neural network
        :param batch_size: size of minibatch for training
        :param epsilon: initial exploration rate
        :param epsilon_decay: decay rate for epsilon
        :param min_epsilon: minimum value for epsilon
        :param target_network_update_freq: frequency of target network updates
        :param replay_memory_size: maximum size of the replay buffer
        :param num_episodes: number of episodes to train for
        :param eval_frequency: frequency of evaluations during training
        :param eval_episodes: number of episodes for each evaluation
        :param train_log_frequency: frequency of logging during training
        :param eval_log_frequency: frequency of logging during evaluation
        :param eval_render: whether to render during evaluation
        :param eval_sleep: time to sleep between steps in evaluation
        :param render: whether to render during training
        :param attacker: whether the agent is an attacker
        :param defender: whether the agent is a defender
        :param save_dir: directory to save results
        :param save_frequency: frequency to save the model during training
        :param hidden_layers: list of hidden layer sizes for the neural network
        """
        self.gamma = gamma
        self.lr = lr
        self.batch_size = batch_size
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.target_network_update_freq = target_network_update_freq
        self.replay_memory_size = replay_memory_size
        self.num_episodes = num_episodes
        self.eval_frequency = eval_frequency
        self.eval_episodes = eval_episodes
        self.train_log_frequency = train_log_frequency
        self.eval_log_frequency = eval_log_frequency
        self.eval_render = eval_render
        self.eval_sleep = eval_sleep
        self.render = render
        self.attacker = attacker
        self.defender = defender
        self.save_dir = save_dir
        self.save_frequency = save_frequency
        self.hidden_layers = hidden_layers


class AgentResult:
    """
    Class to store agent results
    """
    def __init__(self):
        self.episode_rewards = []
        self.episode_steps = []
        self.action_counts = {}
        self.episode_avg_attacker_rewards = []
        self.episode_avg_defender_rewards = []


class DDQNAgent:
    """
    Double Deep Q-Network (DDQN) Agent for gym-idsgame
    """
    def __init__(self, env, config: DDQNConfig):
        """
        Initialize the DDQN agent
        
        :param env: the gym environment
        :param config: configuration for the agent
        """
        self.env = env
        self.config = config
        self.train_result = AgentResult()
        self.eval_result = AgentResult()
        self.state_dim = self.get_state_dim()
        self.num_actions = self.get_num_actions()
        
        # Create main and target networks
        self.policy_network = DDQNNetwork(self.state_dim, self.num_actions, config.hidden_layers, name="policy_network")
        self.target_network = DDQNNetwork(self.state_dim, self.num_actions, config.hidden_layers, name="target_network")
        
        # Initialize networks with a dummy forward pass
        dummy_state = np.zeros((1, self.state_dim), dtype=np.float32)
        self.policy_network(dummy_state)
        self.target_network(dummy_state)
        
        # Initialize target network with policy network weights
        self.update_target_network()
        
        # Replay buffer
        self.replay_buffer = ReplayBuffer(config.replay_memory_size)
        
        # Optimizer
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=config.lr)
        
        # Metrics and logging
        self.train_episode = 0
        self.episode_rewards = []
        self.avg_episode_rewards = []
        self.epsilon = config.epsilon
        
        # Initialize action counts
        for i in range(self.num_actions):
            self.train_result.action_counts[i] = 0
            self.eval_result.action_counts[i] = 0
        
        # Create directory for results
        self.result_dir = config.save_dir
        if not os.path.exists(self.result_dir):
            os.makedirs(self.result_dir)
        
        # Setup logging
        self.log = self.setup_logger()
        
    def get_state_dim(self):
        """
        Get the dimension of the state space
        
        :return: state dimension
        """
        try:
            # For gym-idsgame environment
            if hasattr(self.env, 'idsgame_config'):
                if self.config.attacker:
                    return self.env.idsgame_config.game_config.num_attack_types * self.env.idsgame_config.game_config.num_nodes
                else:
                    return self.env.idsgame_config.game_config.num_defense_types * self.env.idsgame_config.game_config.num_nodes
            else:
                # General case for other gym environments
                return self.env.observation_space.shape[0]
        except Exception as e:
            self.log.error(f"Error determining state dimension: {str(e)}")
            if hasattr(self.env, 'observation_space'):
                if hasattr(self.env.observation_space, 'shape'):
                    return self.env.observation_space.shape[0]
                else:
                    return self.env.observation_space.n
            else:
                raise ValueError("Could not determine state dimension")
    
    def get_num_actions(self):
        """
        Get the dimension of the action space
        
        :return: action dimension
        """
        try:
            # For gym-idsgame environment
            if hasattr(self.env, 'idsgame_config'):
                if self.config.attacker:
                    return self.env.idsgame_config.game_config.num_attack_types * self.env.idsgame_config.game_config.num_nodes
                else:
                    return self.env.idsgame_config.game_config.num_defense_types * self.env.idsgame_config.game_config.num_nodes
            else:
                # General case for other gym environments
                if hasattr(self.env.action_space, 'n'):
                    return self.env.action_space.n
                else:
                    return self.env.action_space.shape[0]
        except Exception as e:
            self.log.error(f"Error determining action dimension: {str(e)}")
            if hasattr(self.env, 'action_space'):
                if hasattr(self.env.action_space, 'n'):
                    return self.env.action_space.n
                else:
                    return self.env.action_space.shape[0]
            else:
                raise ValueError("Could not determine action dimension")
    
    def setup_logger(self):
        """
        Set up the logger
        
        :return: the logger
        """
        logger = logging.getLogger('DDQN_Agent')
        logger.setLevel(logging.INFO)
        # Create file handler
        file_handler = logging.FileHandler(os.path.join(self.result_dir, 'agent.log'))
        file_handler.setLevel(logging.INFO)
        # Create console handler and set level
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        # Create formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        # Add handlers to logger
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        return logger
    
    def update_target_network(self):
        """
        Update the target network with the policy network weights
        """
        self.target_network.set_weights(self.policy_network.get_weights())
    
    def select_action(self, state):
        """
        Select an action based on the current state using epsilon-greedy policy
        
        :param state: current state
        :return: selected action
        """
        if np.random.random() < self.epsilon:
            # Exploration: select random action
            return np.random.randint(self.num_actions)
        else:
            # Exploitation: select best action from Q-network
            state_tensor = tf.convert_to_tensor(state, dtype=tf.float32)
            state_tensor = tf.expand_dims(state_tensor, 0)  # Add batch dimension
            q_values = self.policy_network(state_tensor)
            return tf.argmax(q_values[0]).numpy()
    
    def train(self):
        """
        Train the DDQN agent
        
        :return: training results
        """
        # Initialize training metrics
        self.train_result = AgentResult()
        self.log.info("Starting training")
        
        # Training loop
        episode = 0
        while episode < self.config.num_episodes:
            start_time = time.time()
            
            # Reset environment
            state = self.env.reset()
            episode_reward = 0
            episode_steps = 0
            done = False
            
            # Check if state is a tuple (observation, info) as in Gym >= 0.26.0
            if isinstance(state, tuple):
                state = state[0]
            
            # Collect experience and train
            while not done:
                # Render environment if configured
                if self.config.render:
                    self.env.render()
                
                # Select action using epsilon-greedy
                action = self.select_action(state)
                
                # Take action and observe next state and reward
                next_state, reward, done, info = self.env.step(action)
                
                # Handle different gym versions
                if isinstance(info, dict):
                    if 'TimeLimit.truncated' in info and info['TimeLimit.truncated']:
                        # Environment was artificially terminated due to time limit
                        # We don't want to treat this as a done signal for learning
                        done = False
                
                # Store transition in replay buffer
                self.replay_buffer.push(state, action, reward, next_state, done)
                
                # Update state and metrics
                state = next_state
                episode_reward += reward
                episode_steps += 1
                self.train_result.action_counts[action] += 1
                
                # Train if enough samples in replay buffer
                if len(self.replay_buffer) > self.config.batch_size:
                    self.train_step()
                
                # Check if max steps per episode reached
                if hasattr(self.env, 'idsgame_config') and episode_steps >= self.env.idsgame_config.game_config.max_steps:
                    done = True
            
            # Decay epsilon
            self.epsilon = max(self.config.min_epsilon, 
                              self.epsilon * self.config.epsilon_decay)
            
            # Update episode counter and metrics
            episode += 1
            self.train_episode = episode
            self.episode_rewards.append(episode_reward)
            self.train_result.episode_rewards.append(episode_reward)
            self.train_result.episode_steps.append(episode_steps)
            
            # Save avg reward over last 100 episodes
            if len(self.episode_rewards) > 100:
                avg_reward = sum(self.episode_rewards[-100:]) / 100
            else:
                avg_reward = sum(self.episode_rewards) / len(self.episode_rewards)
            self.avg_episode_rewards.append(avg_reward)
            
            # Update target network periodically
            if episode % self.config.target_network_update_freq == 0:
                self.update_target_network()
                self.log.info("Updated target network")
            
            # Log progress
            if episode % self.config.train_log_frequency == 0:
                self.log.info("Episode: {}, Reward: {}, Avg Reward (100 ep): {:.2f}, Epsilon: {:.2f}, Steps: {}, Time: {:.2f}s".format(
                    episode, episode_reward, avg_reward, self.epsilon, episode_steps, time.time() - start_time))
            
            # Save model periodically
            if episode % self.config.save_frequency == 0:
                self.save_model()
            
            # Evaluate agent periodically
            if episode % self.config.eval_frequency == 0:
                self.eval_model()
        
        # Final save
        self.save_model()
        
        # Final evaluation
        self.eval_model()
        
        return self.train_result
    
    def train_step(self):
        """
        Perform a single training step (experience replay)
        """
        # Sample batch from replay buffer
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.config.batch_size)
        
        # Convert to tensors
        states = tf.convert_to_tensor(states, dtype=tf.float32)
        actions = tf.convert_to_tensor(actions, dtype=tf.int32)
        rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)
        next_states = tf.convert_to_tensor(next_states, dtype=tf.float32)
        dones = tf.convert_to_tensor(dones, dtype=tf.float32)
        
        # Compute Q-values and next Q-values
        with tf.GradientTape() as tape:
            # Q-values for all actions in current states
            q_values = self.policy_network(states)
            
            # Select Q-values for the actions that were actually taken
            actions_one_hot = tf.one_hot(actions, self.num_actions)
            q_values_selected = tf.reduce_sum(q_values * actions_one_hot, axis=1)
            
            # DDQN: Use policy network to select actions and target network to get Q-values
            next_q_values_policy = self.policy_network(next_states)
            next_actions = tf.argmax(next_q_values_policy, axis=1)
            next_actions_one_hot = tf.one_hot(next_actions, self.num_actions)
            
            next_q_values_target = self.target_network(next_states)
            next_q_values_selected = tf.reduce_sum(next_q_values_target * next_actions_one_hot, axis=1)
            
            # Compute target Q-values
            target_q_values = rewards + (1.0 - dones) * self.config.gamma * next_q_values_selected
            
            # Compute loss (Mean Squared Error)
            loss = tf.reduce_mean(tf.square(target_q_values - q_values_selected))
        
        # Compute gradients and apply them
        gradients = tape.gradient(loss, self.policy_network.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.policy_network.trainable_variables))
        
        return loss
    
    def eval_model(self):
        """
        Evaluate the current agent
        """
        self.log.info("Evaluating agent...")
        eval_result = AgentResult()
        
        for i in range(self.config.eval_episodes):
            state = self.env.reset()
            
            # Check if state is a tuple (observation, info) as in Gym >= 0.26.0
            if isinstance(state, tuple):
                state = state[0]
                
            episode_reward = 0
            episode_steps = 0
            done = False
            
            while not done:
                if self.config.eval_render:
                    self.env.render()
                if self.config.eval_sleep:
                    time.sleep(self.config.eval_sleep)
                
                # Always select best action in evaluation (no exploration)
                state_tensor = tf.convert_to_tensor(state, dtype=tf.float32)
                state_tensor = tf.expand_dims(state_tensor, 0)
                q_values = self.policy_network(state_tensor)
                action = tf.argmax(q_values[0]).numpy()
                
                next_state, reward, done, _ = self.env.step(action)
                
                state = next_state
                episode_reward += reward
                episode_steps += 1
                if action in eval_result.action_counts:
                    eval_result.action_counts[action] += 1
                else:
                    eval_result.action_counts[action] = 1
                
                if hasattr(self.env, 'idsgame_config') and episode_steps >= self.env.idsgame_config.game_config.max_steps:
                    done = True
            
            eval_result.episode_rewards.append(episode_reward)
            eval_result.episode_steps.append(episode_steps)
        
        # Calculate average metrics
        avg_reward = sum(eval_result.episode_rewards) / len(eval_result.episode_rewards)
        avg_steps = sum(eval_result.episode_steps) / len(eval_result.episode_steps)
        
        self.log.info("Evaluation: Avg Reward: {:.2f}, Avg Steps: {:.2f}".format(avg_reward, avg_steps))
        self.eval_result = eval_result
        
        return eval_result
    
    def save_model(self):
        """
        Save the model to disk
        """
        save_path = os.path.join(self.result_dir, "model")
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        
        # Save policy network
        policy_path = os.path.join(save_path, "policy_network")
        self.policy_network.save_weights(policy_path)
        
        # Save target network
        target_path = os.path.join(save_path, "target_network")
        self.target_network.save_weights(target_path)
        
        # Save agent metrics
        metrics = {
            "epsilon": self.epsilon,
            "episode_rewards": self.episode_rewards,
            "avg_episode_rewards": self.avg_episode_rewards,
            "train_episode": self.train_episode
        }
        np.save(os.path.join(save_path, "metrics.npy"), metrics)
        
        self.log.info("Model saved to: {}".format(save_path))
    
    def load_model(self, load_path):
        """
        Load the model from disk
        
        :param load_path: path to load the model from
        """
        # Load policy network
        policy_path = os.path.join(load_path, "policy_network")
        self.policy_network.load_weights(policy_path)
        
        # Load target network
        target_path = os.path.join(load_path, "target_network")
        self.target_network.load_weights(target_path)
        
        # Load agent metrics
        metrics = np.load(os.path.join(load_path, "metrics.npy"), allow_pickle=True).item()
        self.epsilon = metrics["epsilon"]
        self.episode_rewards = metrics["episode_rewards"]
        self.avg_episode_rewards = metrics["avg_episode_rewards"]
        self.train_episode = metrics["train_episode"]
        
        self.log.info("Model loaded from: {}".format(load_path))


def create_ddqn_agent(env, config=None):
    """
    Create a DDQN agent with default or custom configuration
    
    :param env: the gym environment
    :param config: configuration for the agent (optional)
    :return: initialized DDQN agent
    """
    if config is None:
        # Use default configuration
        config = DDQNConfig()
    
    # Create agent
    agent = DDQNAgent(env, config)
    
    return agent










# import numpy as np
# import tensorflow as tf
# import random
# from collections import deque
# import os
# import datetime
# import time
# import logging
# # import gym


# class ReplayBuffer:
#     """
#     Replay Buffer for storing transitions for training a DDQN agent
#     """
#     def __init__(self, max_size):
#         """
#         Initialize the replay buffer
        
#         :param max_size: maximum size of the buffer
#         """
#         self.buffer = deque(maxlen=max_size)
    
#     def push(self, state, action, reward, next_state, done):
#         """
#         Add a new transition to the buffer
        
#         :param state: current state
#         :param action: action taken
#         :param reward: reward received
#         :param next_state: next state
#         :param done: if episode is done or not
#         """
#         experience = (state, action, reward, next_state, done)
#         self.buffer.append(experience)
    
#     def sample(self, batch_size):
#         """
#         Sample a batch of experiences from the buffer
        
#         :param batch_size: size of the batch to sample
#         :return: batch of experiences
#         """
#         batch = random.sample(self.buffer, min(len(self.buffer), batch_size))
#         states, actions, rewards, next_states, dones = map(np.stack, zip(*batch))
#         return states, actions, rewards, next_states, dones
    
#     def __len__(self):
#         """
#         Get current size of the buffer
        
#         :return: size of buffer
#         """
#         return len(self.buffer)


# class DDQNNetwork(tf.keras.Model):
#     """
#     Neural Network model for the DDQN agent
#     """
#     def __init__(self, num_states, num_actions, hidden_layers=[64, 64], name="ddqn_network"):
#         """
#         Initialize the network
        
#         :param num_states: dimension of state space
#         :param num_actions: dimension of action space
#         :param hidden_layers: list of hidden layer sizes
#         :param name: name of the network
#         """
#         super(DDQNNetwork, self).__init__(name=name)
        
#         # Define neural network layers
#         self.hidden_layers = []
#         for size in hidden_layers:
#             self.hidden_layers.append(tf.keras.layers.Dense(size, activation='relu'))
        
#         self.output_layer = tf.keras.layers.Dense(num_actions, activation=None)
    
#     def call(self, inputs):
#         """
#         Forward pass through the network
        
#         :param inputs: input to the network (state)
#         :return: Q-values for each action
#         """
#         x = inputs
#         for layer in self.hidden_layers:
#             x = layer(x)
        
#         q_values = self.output_layer(x)
#         return q_values


# class DDQNConfig:
#     """
#     Configuration for the DDQN agent
#     """
#     def __init__(self, gamma=0.99, lr=0.00001, batch_size=32, epsilon=1.0, epsilon_decay=0.999,
#                  min_epsilon=0.01, target_network_update_freq=100, replay_memory_size=10000,
#                  num_episodes=10000, eval_frequency=1000, eval_episodes=100, train_log_frequency=100,
#                  eval_log_frequency=1, eval_render=False, eval_sleep=0.0, render=False,
#                  attacker=True, defender=False, save_dir="./results/data", save_frequency=1000,
#                  hidden_layers=[64, 64]):
#         """
#         Initialize the configuration

#         :param gamma: discount factor
#         :param lr: learning rate for the neural network
#         :param batch_size: size of minibatch for training
#         :param epsilon: initial exploration rate
#         :param epsilon_decay: decay rate for epsilon
#         :param min_epsilon: minimum value for epsilon
#         :param target_network_update_freq: frequency of target network updates
#         :param replay_memory_size: maximum size of the replay buffer
#         :param num_episodes: number of episodes to train for
#         :param eval_frequency: frequency of evaluations during training
#         :param eval_episodes: number of episodes for each evaluation
#         :param train_log_frequency: frequency of logging during training
#         :param eval_log_frequency: frequency of logging during evaluation
#         :param eval_render: whether to render during evaluation
#         :param eval_sleep: time to sleep between steps in evaluation
#         :param render: whether to render during training
#         :param attacker: whether the agent is an attacker
#         :param defender: whether the agent is a defender
#         :param save_dir: directory to save results
#         :param save_frequency: frequency to save the model during training
#         :param hidden_layers: list of hidden layer sizes for the neural network
#         """
#         self.gamma = gamma
#         self.lr = lr
#         self.batch_size = batch_size
#         self.epsilon = epsilon
#         self.epsilon_decay = epsilon_decay
#         self.min_epsilon = min_epsilon
#         self.target_network_update_freq = target_network_update_freq
#         self.replay_memory_size = replay_memory_size
#         self.num_episodes = num_episodes
#         self.eval_frequency = eval_frequency
#         self.eval_episodes = eval_episodes
#         self.train_log_frequency = train_log_frequency
#         self.eval_log_frequency = eval_log_frequency
#         self.eval_render = eval_render
#         self.eval_sleep = eval_sleep
#         self.render = render
#         self.attacker = attacker
#         self.defender = defender
#         self.save_dir = save_dir
#         self.save_frequency = save_frequency
#         self.hidden_layers = hidden_layers


# class AgentResult:
#     """
#     Class to store agent results
#     """
#     def __init__(self):
#         self.episode_rewards = []
#         self.episode_steps = []
#         self.action_counts = {}
#         self.episode_avg_attacker_rewards = []
#         self.episode_avg_defender_rewards = []


# class DDQNAgent:
#     """
#     Double Deep Q-Network (DDQN) Agent for gym-idsgame
#     """
#     def __init__(self, env, config: DDQNConfig):
#         """
#         Initialize the DDQN agent
        
#         :param env: the gym environment
#         :param config: configuration for the agent
#         """
#         self.env = env
#         self.config = config
#         self.train_result = AgentResult()
#         self.eval_result = AgentResult()
#         self.state_dim = self.get_state_dim()
#         self.num_actions = self.get_num_actions()
        
#         # Create main and target networks
#         self.policy_network = DDQNNetwork(self.state_dim, self.num_actions, config.hidden_layers, name="policy_network")
#         self.target_network = DDQNNetwork(self.state_dim, self.num_actions, config.hidden_layers, name="target_network")
        
#         # Initialize networks with a dummy forward pass
#         dummy_state = np.zeros((1, self.state_dim), dtype=np.float32)
#         self.policy_network(dummy_state)
#         self.target_network(dummy_state)
        
#         # Initialize target network with policy network weights
#         self.update_target_network()
        
#         # Replay buffer
#         self.replay_buffer = ReplayBuffer(config.replay_memory_size)
        
#         # Optimizer
#         self.optimizer = tf.keras.optimizers.Adam(learning_rate=config.lr)
        
#         # Metrics and logging
#         self.train_episode = 0
#         self.episode_rewards = []
#         self.avg_episode_rewards = []
#         self.epsilon = config.epsilon
        
#         # Initialize action counts
#         for i in range(self.num_actions):
#             self.train_result.action_counts[i] = 0
#             self.eval_result.action_counts[i] = 0
        
#         # Create directory for results
#         self.result_dir = config.save_dir
#         if not os.path.exists(self.result_dir):
#             os.makedirs(self.result_dir)
        
#         # Setup logging
#         self.log = self.setup_logger()
        
#     def get_state_dim(self):
#         """
#         Get the dimension of the state space
        
#         :return: state dimension
#         """
#         try:
#             # For gym-idsgame environment
#             if hasattr(self.env, 'idsgame_config'):
#                 if self.config.attacker:
#                     return self.env.idsgame_config.game_config.num_attack_types * self.env.idsgame_config.game_config.num_nodes
#                 else:
#                     return self.env.idsgame_config.game_config.num_defense_types * self.env.idsgame_config.game_config.num_nodes
#             else:
#                 # General case for other gym environments
#                 return self.env.observation_space.shape[0]
#         except Exception as e:
#             self.log.error(f"Error determining state dimension: {str(e)}")
#             if hasattr(self.env, 'observation_space'):
#                 if hasattr(self.env.observation_space, 'shape'):
#                     return self.env.observation_space.shape[0]
#                 else:
#                     return self.env.observation_space.n
#             else:
#                 raise ValueError("Could not determine state dimension")
    
#     def get_num_actions(self):
#         """
#         Get the dimension of the action space
        
#         :return: action dimension
#         """
#         try:
#             # For gym-idsgame environment
#             if hasattr(self.env, 'idsgame_config'):
#                 if self.config.attacker:
#                     return self.env.idsgame_config.game_config.num_attack_types * self.env.idsgame_config.game_config.num_nodes
#                 else:
#                     return self.env.idsgame_config.game_config.num_defense_types * self.env.idsgame_config.game_config.num_nodes
#             else:
#                 # General case for other gym environments
#                 if hasattr(self.env.action_space, 'n'):
#                     return self.env.action_space.n
#                 else:
#                     return self.env.action_space.shape[0]
#         except Exception as e:
#             self.log.error(f"Error determining action dimension: {str(e)}")
#             if hasattr(self.env, 'action_space'):
#                 if hasattr(self.env.action_space, 'n'):
#                     return self.env.action_space.n
#                 else:
#                     return self.env.action_space.shape[0]
#             else:
#                 raise ValueError("Could not determine action dimension")
    
#     def setup_logger(self):
#         """
#         Set up the logger
        
#         :return: the logger
#         """
#         logger = logging.getLogger('DDQN_Agent')
#         logger.setLevel(logging.INFO)
#         # Create file handler
#         file_handler = logging.FileHandler(os.path.join(self.result_dir, 'agent.log'))
#         file_handler.setLevel(logging.INFO)
#         # Create console handler and set level
#         console_handler = logging.StreamHandler()
#         console_handler.setLevel(logging.INFO)
#         # Create formatter
#         formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
#         file_handler.setFormatter(formatter)
#         console_handler.setFormatter(formatter)
#         # Add handlers to logger
#         logger.addHandler(file_handler)
#         logger.addHandler(console_handler)
#         return logger
    
#     def update_target_network(self):
#         """
#         Update the target network with the policy network weights
#         """
#         self.target_network.set_weights(self.policy_network.get_weights())
    
#     def select_action(self, state):
#         """
#         Select an action based on the current state using epsilon-greedy policy
        
#         :param state: current state
#         :return: selected action
#         """
#         if np.random.random() < self.epsilon:
#             # Exploration: select random action
#             return np.random.randint(self.num_actions)
#         else:
#             # Exploitation: select best action from Q-network
#             state_tensor = tf.convert_to_tensor(state, dtype=tf.float32)
#             state_tensor = tf.expand_dims(state_tensor, 0)  # Add batch dimension
#             q_values = self.policy_network(state_tensor)
#             return tf.argmax(q_values[0]).numpy()
    
#     def train(self):
#         """
#         Train the DDQN agent
        
#         :return: training results
#         """
#         # Initialize training metrics
#         self.train_result = AgentResult()
#         self.log.info("Starting training")
        
#         # Training loop
#         episode = 0
#         while episode < self.config.num_episodes:
#             start_time = time.time()
            
#             # Reset environment
#             state = self.env.reset()
#             episode_reward = 0
#             episode_steps = 0
#             done = False
            
#             # Check if state is a tuple (observation, info) as in Gym >= 0.26.0
#             if isinstance(state, tuple):
#                 state = state[0]
            
#             # Collect experience and train
#             while not done:
#                 # Render environment if configured
#                 if self.config.render:
#                     self.env.render()
                
#                 # Select action using epsilon-greedy
#                 action = self.select_action(state)
                
#                 # Take action and observe next state and reward
#                 next_state, reward, done, info = self.env.step(action)
                
#                 # Handle different gym versions
#                 if isinstance(info, dict):
#                     if 'TimeLimit.truncated' in info and info['TimeLimit.truncated']:
#                         # Environment was artificially terminated due to time limit
#                         # We don't want to treat this as a done signal for learning
#                         done = False
                
#                 # Store transition in replay buffer
#                 self.replay_buffer.push(state, action, reward, next_state, done)
                
#                 # Update state and metrics
#                 state = next_state
#                 episode_reward += reward
#                 episode_steps += 1
#                 self.train_result.action_counts[action] += 1
                
#                 # Train if enough samples in replay buffer
#                 if len(self.replay_buffer) > self.config.batch_size:
#                     self.train_step()
                
#                 # Check if max steps per episode reached
#                 if hasattr(self.env, 'idsgame_config') and episode_steps >= self.env.idsgame_config.game_config.max_steps:
#                     done = True
            
#             # Decay epsilon
#             self.epsilon = max(self.config.min_epsilon, 
#                               self.epsilon * self.config.epsilon_decay)
            
#             # Update episode counter and metrics
#             episode += 1
#             self.train_episode = episode
#             self.episode_rewards.append(episode_reward)
#             self.train_result.episode_rewards.append(episode_reward)
#             self.train_result.episode_steps.append(episode_steps)
            
#             # Save avg reward over last 100 episodes
#             if len(self.episode_rewards) > 100:
#                 avg_reward = sum(self.episode_rewards[-100:]) / 100
#             else:
#                 avg_reward = sum(self.episode_rewards) / len(self.episode_rewards)
#             self.avg_episode_rewards.append(avg_reward)
            
#             # Update target network periodically
#             if episode % self.config.target_network_update_freq == 0:
#                 self.update_target_network()
#                 self.log.info("Updated target network")
            
#             # Log progress
#             if episode % self.config.train_log_frequency == 0:
#                 self.log.info("Episode: {}, Reward: {}, Avg Reward (100 ep): {:.2f}, Epsilon: {:.2f}, Steps: {}, Time: {:.2f}s".format(
#                     episode, episode_reward, avg_reward, self.epsilon, episode_steps, time.time() - start_time))
            
#             # Save model periodically
#             if episode % self.config.save_frequency == 0:
#                 self.save_model()
            
#             # Evaluate agent periodically
#             if episode % self.config.eval_frequency == 0:
#                 self.eval_model()
        
#         # Final save
#         self.save_model()
        
#         # Final evaluation
#         self.eval_model()
        
#         return self.train_result
    
#     def train_step(self):
#         """
#         Perform a single training step (experience replay)
#         """
#         # Sample batch from replay buffer
#         states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.config.batch_size)
        
#         # Convert to tensors
#         states = tf.convert_to_tensor(states, dtype=tf.float32)
#         actions = tf.convert_to_tensor(actions, dtype=tf.int32)
#         rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)
#         next_states = tf.convert_to_tensor(next_states, dtype=tf.float32)
#         dones = tf.convert_to_tensor(dones, dtype=tf.float32)
        
#         # Compute Q-values and next Q-values
#         with tf.GradientTape() as tape:
#             # Q-values for all actions in current states
#             q_values = self.policy_network(states)
            
#             # Select Q-values for the actions that were actually taken
#             actions_one_hot = tf.one_hot(actions, self.num_actions)
#             q_values_selected = tf.reduce_sum(q_values * actions_one_hot, axis=1)
            
#             # DDQN: Use policy network to select actions and target network to get Q-values
#             next_q_values_policy = self.policy_network(next_states)
#             next_actions = tf.argmax(next_q_values_policy, axis=1)
#             next_actions_one_hot = tf.one_hot(next_actions, self.num_actions)
            
#             next_q_values_target = self.target_network(next_states)
#             next_q_values_selected = tf.reduce_sum(next_q_values_target * next_actions_one_hot, axis=1)
            
#             # Compute target Q-values
#             target_q_values = rewards + (1.0 - dones) * self.config.gamma * next_q_values_selected
            
#             # Compute loss (Mean Squared Error)
#             loss = tf.reduce_mean(tf.square(target_q_values - q_values_selected))
        
#         # Compute gradients and apply them
#         gradients = tape.gradient(loss, self.policy_network.trainable_variables)
#         self.optimizer.apply_gradients(zip(gradients, self.policy_network.trainable_variables))
        
#         return loss
    
#     def eval_model(self):
#         """
#         Evaluate the current agent
#         """
#         self.log.info("Evaluating agent...")
#         eval_result = AgentResult()
        
#         for i in range(self.config.eval_episodes):
#             state = self.env.reset()
            
#             # Check if state is a tuple (observation, info) as in Gym >= 0.26.0
#             if isinstance(state, tuple):
#                 state = state[0]
                
#             episode_reward = 0
#             episode_steps = 0
#             done = False
            
#             while not done:
#                 if self.config.eval_render:
#                     self.env.render()
#                 if self.config.eval_sleep:
#                     time.sleep(self.config.eval_sleep)
                
#                 # Always select best action in evaluation (no exploration)
#                 state_tensor = tf.convert_to_tensor(state, dtype=tf.float32)
#                 state_tensor = tf.expand_dims(state_tensor, 0)
#                 q_values = self.policy_network(state_tensor)
#                 action = tf.argmax(q_values[0]).numpy()
                
#                 next_state, reward, done, _ = self.env.step(action)
                
#                 state = next_state
#                 episode_reward += reward
#                 episode_steps += 1
#                 if action in eval_result.action_counts:
#                     eval_result.action_counts[action] += 1
#                 else:
#                     eval_result.action_counts[action] = 1
                
#                 if hasattr(self.env, 'idsgame_config') and episode_steps >= self.env.idsgame_config.game_config.max_steps:
#                     done = True
            
#             eval_result.episode_rewards.append(episode_reward)
#             eval_result.episode_steps.append(episode_steps)
        
#         # Calculate average metrics
#         avg_reward = sum(eval_result.episode_rewards) / len(eval_result.episode_rewards)
#         avg_steps = sum(eval_result.episode_steps) / len(eval_result.episode_steps)
        
#         self.log.info("Evaluation: Avg Reward: {:.2f}, Avg Steps: {:.2f}".format(avg_reward, avg_steps))
#         self.eval_result = eval_result
        
#         return eval_result
    
#     def save_model(self):
#         """
#         Save the model to disk
#         """
#         save_path = os.path.join(self.result_dir, "model")
#         if not os.path.exists(save_path):
#             os.makedirs(save_path)
        
#         # Save policy network
#         policy_path = os.path.join(save_path, "policy_network")
#         self.policy_network.save_weights(policy_path)
        
#         # Save target network
#         target_path = os.path.join(save_path, "target_network")
#         self.target_network.save_weights(target_path)
        
#         # Save agent metrics
#         metrics = {
#             "epsilon": self.epsilon,
#             "episode_rewards": self.episode_rewards,
#             "avg_episode_rewards": self.avg_episode_rewards,
#             "train_episode": self.train_episode
#         }
#         np.save(os.path.join(save_path, "metrics.npy"), metrics)
        
#         self.log.info("Model saved to: {}".format(save_path))
    
#     def load_model(self, load_path):
#         """
#         Load the model from disk
        
#         :param load_path: path to load the model from
#         """
#         # Load policy network
#         policy_path = os.path.join(load_path, "policy_network")
#         self.policy_network.load_weights(policy_path)
        
#         # Load target network
#         target_path = os.path.join(load_path, "target_network")
#         self.target_network.load_weights(target_path)
        
#         # Load agent metrics
#         metrics = np.load(os.path.join(load_path, "metrics.npy"), allow_pickle=True).item()
#         self.epsilon = metrics["epsilon"]
#         self.episode_rewards = metrics["episode_rewards"]
#         self.avg_episode_rewards = metrics["avg_episode_rewards"]
#         self.train_episode = metrics["train_episode"]
        
#         self.log.info("Model loaded from: {}".format(load_path))


# def create_ddqn_agent(env, config=None):
#     """
#     Create a DDQN agent with default or custom configuration
    
#     :param env: the gym environment
#     :param config: configuration for the agent (optional)
#     :return: initialized DDQN agent
#     """
#     if config is None:
#         # Use default configuration
#         config = DDQNConfig()
    
#     # Create agent
#     agent = DDQNAgent(env, config)
    
#     return agent