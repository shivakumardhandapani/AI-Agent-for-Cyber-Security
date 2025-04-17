import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

def plot_training_evaluation_qtable(train_path, eval_path, q_table_path):
    # Load training data
    if os.path.exists(train_path):
        train_df = pd.read_csv(train_path)
        print(f"✅ Training data loaded from: {train_path}")
    else:
        print(f"❌ Training file not found: {train_path}")
        return

    # Load evaluation data
    if os.path.exists(eval_path):
        eval_df = pd.read_csv(eval_path)
        print(f"✅ Evaluation data loaded from: {eval_path}")
    else:
        print(f"❌ Evaluation file not found: {eval_path}")
        return

    # Load Q-table
    if os.path.exists(q_table_path):
        q_table = np.load(q_table_path)
        print(f"✅ Q-table loaded from: {q_table_path}")
    else:
        print(f"❌ Q-table file not found: {q_table_path}")
        return

    # Plot defender reward
    plt.figure(figsize=(10, 4))
    plt.plot(train_df['avg_defender_episode_rewards'], label='Training')
    plt.plot(eval_df['avg_defender_episode_rewards'], label='Evaluation', linestyle='--')
    plt.title("Defender Episode Rewards Over Time")
    plt.xlabel("Logged Point")
    plt.ylabel("Avg Defender Reward")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Plot epsilon decay
    plt.figure(figsize=(10, 4))
    plt.plot(train_df['epsilon_values'], color='green')
    plt.title("Epsilon Decay Over Training")
    plt.xlabel("Logged Point")
    plt.ylabel("Epsilon")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Plot hack probability
    plt.figure(figsize=(10, 4))
    plt.plot(train_df['hack_probability'], label='Training')
    plt.plot(eval_df['hack_probability'], label='Evaluation', linestyle='--')
    plt.title("Hack Probability Over Time")
    plt.xlabel("Logged Point")
    plt.ylabel("Hack Probability")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Plot Q-table heatmap (single state)
    if q_table.shape[0] == 1:
        plt.figure(figsize=(12, 1))
        plt.imshow(q_table, cmap='viridis', aspect='auto')
        plt.colorbar(label='Q-value')
        plt.title("Q-table Heatmap (Single State)")
        plt.yticks([])
        plt.xlabel("Actions")
        plt.tight_layout()
        plt.show()
    else:
        print(f"ℹ️ Q-table has shape {q_table.shape}. Only visualizing if there's one state.")


plot_training_evaluation_qtable(
    train_path=os.path.join(sarsa_config.save_dir, "1744398263.3432512_train_results_checkpoint.csv"),
    eval_path=os.path.join(sarsa_config.save_dir, "1744398263.3432512_eval_results_checkpoint.csv"),
    q_table_path=os.path.join(sarsa_config.save_dir, "1744398263.342431_defender_q_table.npy")
)