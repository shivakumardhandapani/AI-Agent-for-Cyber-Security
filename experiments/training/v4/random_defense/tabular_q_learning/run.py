import os
import time
import sys
from gym_idsgame.config.runner_mode import RunnerMode
from gym_idsgame.agents.training_agents.q_learning.q_agent_config import QAgentConfig
from gym_idsgame.agents.dao.agent_type import AgentType
from gym_idsgame.config.client_config import ClientConfig
from gym_idsgame.runnner import Runner
from experiments.util import plotting_util, util


def get_script_path():
    """
    :return: the script path
    """
    return os.path.dirname(os.path.realpath(sys.argv[0]))


def default_output_dir() -> str:
    """
    :return: the default output dir
    """
    script_dir = get_script_path()
    return script_dir


def default_config_path() -> str:
    """
    :return: the default path to configuration file
    """
    config_path = os.path.join(default_output_dir(), './config.json')
    return config_path


def default_config() -> ClientConfig:
    """
    :return: Default configuration for the experiment
    """
    q_agent_config = QAgentConfig(gamma=0.999, alpha=0.05, epsilon=1, render=False, eval_sleep=0.9,
                                  min_epsilon=0.01, eval_episodes=100, train_log_frequency=1,
                                  epsilon_decay=0.9999, video=True, eval_log_frequency=1,
                                  video_fps=5, video_dir=default_output_dir() + "/videos", num_episodes=40000,
                                  eval_render=False, gifs=True, gif_dir=default_output_dir() + "/gifs",
                                  eval_frequency=5000, attacker=True, defender=False,
                                  video_frequency=101,
                                  save_dir=default_output_dir() + "/data")
    env_name = "idsgame-random_defense-v4"
    client_config = ClientConfig(env_name=env_name, attacker_type=AgentType.TABULAR_Q_AGENT.value,
                                 mode=RunnerMode.TRAIN_ATTACKER.value,
                                 q_agent_config=q_agent_config, output_dir=default_output_dir(),
                                 title="TrainingQAgent vs RandomDefender")
    return client_config


def write_default_config(path:str = None) -> None:
    """
    Writes the default configuration to a json file

    :param path: the path to write the configuration to
    :return: None
    """
    if path is None:
        path = default_config_path()
    config = default_config()
    util.write_config_file(config, path)


def plot_csv(config: ClientConfig, eval_csv_path:str, train_csv_path: str) -> None:
    """
    Plot results from csv files

    :param config: client config
    :param eval_csv_path: path to the csv file with evaluation results
    :param train_csv_path: path to the csv file with training results
    :return: None
    """
    plotting_util.read_and_plot_results(train_csv_path, eval_csv_path, config.q_agent_config.train_log_frequency,
                                        config.q_agent_config.eval_frequency, config.q_agent_config.eval_log_frequency,
                                        config.q_agent_config.eval_episodes, config.output_dir, sim=False)


# Program entrypoint
if __name__ == '__main__':
    args = util.parse_args(default_config_path())
    if args.configpath is not None:
        if not os.path.exists(args.configpath):
            write_default_config()
        config = util.read_config(args.configpath)
    else:
        config = default_config()
    time_str = str(time.time())
    util.create_artefact_dirs(config.output_dir)
    logger = util.setup_logger("tabular_q_learning_vs_random_defense-v4", config.output_dir + "/logs/",
                               time_str=time_str)
    config.logger = logger
    config.q_agent_config.logger = logger
    config.q_agent_config.to_csv(config.output_dir + "/hyperparameters/" + time_str + ".csv")
    train_result, eval_result = Runner.run(config)
    if len(train_result.avg_episode_steps) > 0 and len(eval_result.avg_episode_steps) > 0:
        train_csv_path = config.output_dir + "/data/" + time_str + "_train" + ".csv"
        train_result.to_csv(train_csv_path)
        eval_csv_path = config.output_dir + "/data/" + time_str + "_eval" + ".csv"
        eval_result.to_csv(eval_csv_path)
        plot_csv(config, eval_csv_path, train_csv_path)



