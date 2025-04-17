import os
import gymnasium as gym
import warnings
import numpy
from datetime import datetime

from gym_idsgame.agents.training_agents.q_learning.q_agent_config import QAgentConfig
from experiments.util import util
from experiments.util.plotting_util import read_and_plot_results
from src.agents.sarsa_agent import SARSAAgent
from src.utils.utils import get_output_dir, print_summary
from src.environment.compatibility_wrapper import GymCompatibilityWrapper
from src.utils.plotting import plot_training_evaluation_qtable

%matplotlib inline
warnings.filterwarnings('ignore')

random_seed = 33
output_dir = get_output_dir(algorithm='sarsa')
util.create_artefact_dirs(output_dir, random_seed)

sarsa_config = QAgentConfig(
    gamma=0.995,
    alpha=0.001,
    epsilon=0.95,
    min_epsilon=0.01,
    epsilon_decay=0.9995,

    num_episodes=25000,
    train_log_frequency=200,
    eval_log_frequency=5,
    eval_frequency=750,
    eval_episodes=100,

    attacker=False,
    defender=True,

    render=False,
    eval_render=False,
    video=True,
    gifs=True,
    video_frequency=250,
    video_fps=6,
    video_dir=output_dir + "/results/videos/" + str(random_seed),
    gif_dir=output_dir + "/results/gifs/" + str(random_seed),
    save_dir=output_dir + "/results/data/" + str(random_seed)
)

print("Configuration initialized!")
print(f"Output directory: {output_dir}")

print("\nEnvironment Information:")
