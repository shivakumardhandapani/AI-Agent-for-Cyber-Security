import os
import gym
import gym_idsgame
import numpy as np
import torch
import matplotlib.pyplot as plt
import time
from datetime import datetime
import csv
import random

# Import the properly designed DQN agent
# Note: Save the agent implementation in a file named dqn_agent.py
from dqn_agent1 import DQNAgent

# Helper functions for visualization and tracking
def plot_rewards(rewards, losses=None, epsilons=None, window_size=10, filename="training_progress.png"):
    """Plot training metrics with smoothing"""
    plt.figure(figsize=(12, 8))
    
    # Create a smoothed version of rewards
    if window_size > 0 and len(rewards) > window_size:
        smoothed_rewards = []
        for i in range(len(rewards)):
            start_idx = max(0, i - window_size + 1)
            smoothed_rewards.append(np.mean(rewards[start_idx:i+1]))
    else:
        smoothed_rewards = rewards
    
    # Plot rewards
    plt.subplot(3, 1, 1)
    plt.plot(rewards, alpha=0.4, label='Raw Rewards')
    plt.plot(smoothed_rewards, linewidth=2, label=f'Smoothed (window={window_size})')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Training Rewards')
    plt.legend()
    plt.grid(True)
    
    # Plot epsilon if available
    if epsilons is not None:
        plt.subplot(3, 1, 2)
        plt.plot(epsilons)
        plt.xlabel('Episode')
        plt.ylabel('Epsilon')
        plt.title('Exploration Rate')
        plt.grid(True)
    
    # Plot losses if available
    if losses is not None:
        plt.subplot(3, 1, 3)
        
        # Filter out None values
        filtered_losses = [l for l in losses if l is not None]
        episodes = range(len(filtered_losses))
        
        if window_size > 0 and len(filtered_losses) > window_size:
            smoothed_losses = []
            for i in range(len(filtered_losses)):
                start_idx = max(0, i - window_size + 1)
                smoothed_losses.append(np.mean(filtered_losses[start_idx:i+1]))
        else:
            smoothed_losses = filtered_losses
        
        plt.plot(episodes, filtered_losses, alpha=0.4, label='Loss')
        plt.plot(episodes, smoothed_losses, linewidth=2, label=f'Smoothed Loss')
        plt.xlabel('Episode')
        plt.ylabel('Loss')
        plt.title('Training Loss')
        plt.legend()
        plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def save_metrics(rewards, losses, epsilons, filename="training_metrics.csv"):
    """Save training metrics to CSV"""
    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Episode', 'Reward', 'Loss', 'Epsilon'])
        
        # Handle possible different lengths
        max_len = max(len(rewards), len(losses), len(epsilons))
        
        for i in range(max_len):
            row = [i]
            row.append(rewards[i] if i < len(rewards) else None)
            row.append(losses[i] if i < len(losses) else None)
            row.append(epsilons[i] if i < len(epsilons) else None)
            writer.writerow(row)

def print_summary(agent, rewards, steps, duration):
    """Print a summary of training results"""
    print("\n========== TRAINING SUMMARY ==========")
    print(f"Total Training Time: {duration:.2f} seconds")
    print(f"Episodes Completed: {len(rewards)}")
    print(f"Total Steps: {agent.steps}")
    
    if len(rewards) > 0:
        print(f"Average Reward (all): {np.mean(rewards):.2f}")
        print(f"Average Reward (last 100): {np.mean(rewards[-100:]):.2f}")
        print(f"Max Reward: {np.max(rewards):.2f}")
        print(f"Min Reward: {np.min(rewards):.2f}")
    
    if len(steps) > 0:
        print(f"Average Episode Length: {np.mean(steps):.2f} steps")
    
    print(f"Final Exploration Rate (epsilon): {agent.epsilon:.4f}")
    
    # Print reward statistics
    stats = agent.get_stats()
    if 'reward_stats' in stats:
        print("\nReward Statistics:")
        for key, value in stats['reward_stats'].items():
            print(f"  {key}: {value:.4f}")

def extract_attacker_obs(obs):
    """Extract and flatten attacker observations"""
    obs = obs[0] if isinstance(obs, tuple) else obs
    return np.array(obs).flatten()

# Main training function
def train(env_name="idsgame-random_attack-v8", 
          num_episodes=1000, 
          max_steps=100,
          save_interval=100,
          plot_interval=50,
          random_seed=42,
          output_dir="dqn_results"):
    """
    Train a DQN agent on the IDS game environment
    
    Args:
        env_name: Name of the gym environment
        num_episodes: Number of training episodes
        max_steps: Maximum steps per episode
        save_interval: Save the model every N episodes
        plot_interval: Plot progress every N episodes
        random_seed: Random seed for reproducibility 
        output_dir: Directory to save results
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Set random seeds for reproducibility
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    
    # Initialize environment
    env = gym.make(env_name)
    
    # Print environment information
    print("\nEnvironment Information:")
    print(f"Observation Space: {env.observation_space}")
    print(f"Action Space: {env.action_space}")
    
    # Get dimensions for the agent
    sample_obs = env.reset()[0]
    state_dim = np.array(sample_obs).flatten().shape[0]
    action_dim = env.attacker_action_space.n
    
    print(f"State Dimension: {state_dim}")
    print(f"Action Dimension: {action_dim}")
    
    # Initialize agent
    agent = DQNAgent(
        state_dim=state_dim, 
        action_dim=action_dim,
        buffer_capacity=50000,  # Larger buffer
        gamma=0.99,
        lr=0.001,
        batch_size=64,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=0.995,
        tau=0.005
    )
    
    # Training metrics
    all_rewards = []
    all_losses = []
    all_epsilons = []
    episode_steps = []
    
    # Start training
    start_time = time.time()
    print(f"\nStarting training for {num_episodes} episodes...")
    
    for episode in range(1, num_episodes + 1):
        episode_start = time.time()
        state = extract_attacker_obs(env.reset())
        episode_reward = 0
        total_loss = 0
        update_count = 0
        step = 0
        
        # Run one episode
        done = False
        while not done and step < max_steps:
            # Select attacker action
            action = agent.select_action(state)
            
            # Random defender action (for simplicity)
            def_action = env.defender_action_space.sample()
            full_action = (action, def_action)
            
            # Take step in environment
            next_obs, reward, terminated, truncated, _ = env.step(full_action)
            done = terminated or truncated
            next_state = extract_attacker_obs(next_obs)
            
            # Store transition with original reward
            agent.store(state, action, reward[0], next_state, done)
            
            # Update agent
            loss = agent.update()
            if loss is not None:
                total_loss += loss
                update_count += 1
            
            # Update state and tracking
            state = next_state
            episode_reward += reward[0]  # Use original reward for tracking
            step += 1
        
        # End of episode processing
        agent.end_episode(episode_reward)
        episode_steps.append(step)
        
        # Store metrics
        all_rewards.append(episode_reward)
        all_epsilons.append(agent.epsilon)
        all_losses.append(total_loss / max(1, update_count))
        
        # Calculate statistics for logging
        elapsed = time.time() - episode_start
        rewards_window = all_rewards[-10:] if len(all_rewards) >= 10 else all_rewards
        avg_reward = np.mean(rewards_window)
        
        # Logging
        if episode % 10 == 0 or episode == 1:
            print(f"Episode {episode}/{num_episodes} | " +
                  f"Reward: {episode_reward:.2f} | " +
                  f"Avg(10): {avg_reward:.2f} | " +
                  f"Steps: {step} | " +
                  f"Epsilon: {agent.epsilon:.3f} | " +
                  f"Time: {elapsed:.2f}s")
        
        # Save model periodically
        if episode % save_interval == 0 or episode == num_episodes:
            agent.save(f"{output_dir}/dqn_model_ep{episode}.pt")
        
        # Plot progress periodically
        if episode % plot_interval == 0 or episode == num_episodes:
            plot_rewards(
                all_rewards, 
                all_losses, 
                all_epsilons, 
                window_size=10,
                filename=f"{output_dir}/training_progress_ep{episode}.png"
            )
            save_metrics(
                all_rewards,
                all_losses,
                all_epsilons,
                filename=f"{output_dir}/metrics_ep{episode}.csv"
            )
    
    # Training complete
    total_time = time.time() - start_time
    print(f"\nTraining completed in {total_time:.2f} seconds")
    
    # Save final model and metrics
    agent.save(f"{output_dir}/dqn_model_final.pt")
    plot_rewards(all_rewards, all_losses, all_epsilons, 
                 filename=f"{output_dir}/final_training_progress.png")
    save_metrics(all_rewards, all_losses, all_epsilons, 
                 filename=f"{output_dir}/final_metrics.csv")
    
    # Print summary
    print_summary(agent, all_rewards, episode_steps, total_time)
    
    return agent, all_rewards

# Evaluation function
def evaluate(agent, env_name="idsgame-random_attack-v8", num_episodes=20, max_steps=100, render=False):
    """
    Evaluate a trained DQN agent
    
    Args:
        agent: Trained DQN agent
        env_name: Name of the environment
        num_episodes: Number of evaluation episodes
        max_steps: Maximum steps per episode
        render: Whether to render the environment
    
    Returns:
        dict: Evaluation results
    """
    env = gym.make(env_name)
    
    # Set agent to evaluation mode
    agent.epsilon = 0.0  # No exploration during evaluation
    
    all_rewards = []
    all_steps = []
    success_count = 0
    
    print("\nStarting evaluation...")
    
    for episode in range(1, num_episodes + 1):
        state = extract_attacker_obs(env.reset())
        episode_reward = 0
        step = 0
        done = False
        
        while not done and step < max_steps:
            # Select action (greedy policy)
            action = agent.select_action(state)
            
            # Random defender action
            def_action = env.defender_action_space.sample()
            full_action = (action, def_action)
            
            # Take step
            next_obs, reward, terminated, truncated, _ = env.step(full_action)
            done = terminated or truncated
            next_state = extract_attacker_obs(next_obs)
            
            # Update
            state = next_state
            episode_reward += reward[0]
            step += 1
            
            # Optional rendering
            if render:
                env.render()
        
        # Episode complete
        all_rewards.append(episode_reward)
        all_steps.append(step)
        
        # Count as success if reward is positive
        if episode_reward > 0:
            success_count += 1
        
        print(f"Eval Episode {episode}/{num_episodes} | " +
              f"Reward: {episode_reward:.2f} | " +
              f"Steps: {step}")
    
    # Calculate statistics
    avg_reward = np.mean(all_rewards)
    std_reward = np.std(all_rewards)
    min_reward = np.min(all_rewards)
    max_reward = np.max(all_rewards)
    avg_steps = np.mean(all_steps)
    success_rate = (success_count / num_episodes) * 100
    
    # Print results
    print("\n========== EVALUATION RESULTS ==========")
    print(f"Episodes: {num_episodes}")
    print(f"Average Reward: {avg_reward:.2f} ± {std_reward:.2f}")
    print(f"Min/Max Reward: {min_reward:.2f} / {max_reward:.2f}")
    print(f"Average Steps: {avg_steps:.2f}")
    print(f"Success Rate: {success_rate:.2f}%")
    
    return {
        "avg_reward": avg_reward,
        "std_reward": std_reward,
        "min_reward": min_reward,
        "max_reward": max_reward,
        "avg_steps": avg_steps,
        "success_rate": success_rate,
        "all_rewards": all_rewards,
        "all_steps": all_steps
    }

# Run training and evaluation if script is executed directly
if __name__ == "__main__":
    # Training parameters
    ENV_NAME = "idsgame-random_attack-v8"
    NUM_EPISODES = 1000
    MAX_STEPS = 100
    RANDOM_SEED = 42
    OUTPUT_DIR = "dqn_results"
    
    # Train agent
    trained_agent, _ = train(
        env_name=ENV_NAME,
        num_episodes=NUM_EPISODES,
        max_steps=MAX_STEPS,
        random_seed=RANDOM_SEED,
        output_dir=OUTPUT_DIR
    )
    
    # Evaluate agent
    evaluate(
        agent=trained_agent,
        env_name=ENV_NAME,
        num_episodes=20,
        max_steps=MAX_STEPS
    )