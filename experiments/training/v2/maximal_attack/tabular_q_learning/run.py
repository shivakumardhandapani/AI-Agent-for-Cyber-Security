import os
import time
import sys
from gym_idsgame.config.runner_mode import RunnerMode
from gym_idsgame.agents.training_agents.q_learning.q_agent_config import QAgentConfig
from gym_idsgame.agents.dao.agent_type import AgentType
from gym_idsgame.config.client_config import ClientConfig
from gym_idsgame.runnner import Runner
from experiments.util import plotting_util, util
import glob

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
    q_agent_config = QAgentConfig(gamma=0.999, alpha=0.0005, epsilon=1, render=False, eval_sleep=0.9,
                                  min_epsilon=0.01, eval_episodes=100, train_log_frequency=100,
                                  epsilon_decay=0.9999, video=True, eval_log_frequency=1,
                                  video_fps=5, video_dir=default_output_dir() + "/results/videos", num_episodes=20001,
                                  eval_render=False, gifs=True, gif_dir=default_output_dir() + "/results/gifs",
                                  eval_frequency=1000, attacker=False, defender=True,
                                  video_frequency=101,
                                  save_dir=default_output_dir() + "/results/data")
    env_name = "idsgame-maximal_attack-v2"
    client_config = ClientConfig(env_name=env_name, defender_type=AgentType.TABULAR_Q_AGENT.value,
                                 mode=RunnerMode.TRAIN_DEFENDER.value,
                                 q_agent_config=q_agent_config, output_dir=default_output_dir(),
                                 title="AttackMaximalAttacker vs TrainingQAgent",
                                 run_many=True, random_seeds=[0, 999, 299, 399, 499])
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


def plot_csv(config: ClientConfig, eval_csv_path:str, train_csv_path: str, attack_stats_csv_path : str = None,
             random_seed : int = 0) -> None:
    """
    Plot results from csv files

    :param config: client config
    :param eval_csv_path: path to the csv file with evaluation results
    :param train_csv_path: path to the csv file with training results
    :param random_seed: the random seed of the experiment
    :param attack_stats_csv_path: path to attack stats
    :return: None
    """
    plotting_util.read_and_plot_results(train_csv_path, eval_csv_path,
                                        config.q_agent_config.train_log_frequency,
                                        config.q_agent_config.eval_frequency, config.q_agent_config.eval_log_frequency,
                                        config.q_agent_config.eval_episodes, config.output_dir, sim=False,
                                        random_seed = random_seed, attack_stats_csv_path = attack_stats_csv_path)


def plot_average_results(experiment_title :str, config: ClientConfig, eval_csv_paths:list,
                         train_csv_paths: str) -> None:
    """
    Plots average results after training with different seeds

    :param experiment_title: title of the experiment
    :param config: experiment config
    :param eval_csv_paths: paths to csv files with evaluation data
    :param train_csv_paths: path to csv files with training data
    :return: None
    """
    plotting_util.read_and_plot_average_results(experiment_title, train_csv_paths, eval_csv_paths,
                                                config.q_agent_config.train_log_frequency,
                                                config.q_agent_config.eval_frequency,
                                                config.output_dir)


def run_experiment(configpath: str, random_seed: int, noconfig: bool):
    """
    Runs one experiment and saves results and plots

    :param configpath: path to config file
    :param noconfig: whether to override config
    :return: (train_csv_path, eval_csv_path)
    """
    if configpath is not None and not noconfig:
        if not os.path.exists(configpath):
            write_default_config()
        config = util.read_config(configpath)
    else:
        config = default_config()
    time_str = str(time.time())
    util.create_artefact_dirs(config.output_dir, random_seed)
    logger = util.setup_logger("maximal_attack_vs_tabular_q_learning-v2", config.output_dir + "/results/logs/" +
                               str(random_seed) + "/",
                               time_str=time_str)
    config.q_agent_config.save_dir = default_output_dir() + "/results/data/" + str(random_seed) + "/"
    config.q_agent_config.video_dir= default_output_dir() + "/results/videos/" + str(random_seed) + "/"
    config.q_agent_config.gif_dir= default_output_dir() + "/results/gifs/" + str(random_seed) + "/"
    config.logger = logger
    config.q_agent_config.logger = logger
    config.q_agent_config.random_seed = random_seed
    config.random_seed = random_seed
    config.q_agent_config.to_csv(config.output_dir + "/results/hyperparameters/" + str(random_seed) + "/" + time_str + ".csv")
    train_result, eval_result = Runner.run(config)
    train_csv_path = ""
    eval_csv_path = ""
    if len(train_result.avg_episode_steps) > 0 and len(eval_result.avg_episode_steps) > 0:
        train_csv_path = config.output_dir + "/results/data/" + str(random_seed) + "/" + time_str + "_train" + ".csv"
        train_result.to_csv(train_csv_path)
        eval_csv_path = config.output_dir + "/results/data/" + str(random_seed) + "/" + time_str + "_eval" + ".csv"
        eval_result.to_csv(eval_csv_path)
        plot_csv(config, eval_csv_path, train_csv_path, random_seed)

    return train_csv_path, eval_csv_path

# Program entrypoint
if __name__ == '__main__':
    args = util.parse_args(default_config_path())
    experiment_title = "maximal attack vs Q-learning"
    if args.configpath is not None and not args.noconfig:
        if not os.path.exists(args.configpath):
            write_default_config()
        config = util.read_config(args.configpath)
    else:
        config = default_config()
    if args.plotonly:
        base_dir = default_output_dir() + "/results/data/"
        train_csv_paths = []
        eval_csv_paths = []
        if config.run_many:
            for seed in config.random_seeds:
                train_csv_path = glob.glob(base_dir + str(seed) + "/*_train.csv")[0]
                eval_csv_path = glob.glob(base_dir + str(seed) + "/*_eval.csv")[0]
                attack_stats_csv_path = None
                try:
                    attack_stats_csv_paths = glob.glob(base_dir + str(seed) + "/attack_stats_*.csv")
                    attack_stats_csv_path = list(filter(lambda x: "checkpoint" not in attack_stats_csv_paths, attack_stats_csv_paths))[0]
                except:
                    pass
                train_csv_paths.append(train_csv_path)
                eval_csv_paths.append(eval_csv_path)
                plot_csv(config, eval_csv_path, train_csv_path, attack_stats_csv_path, random_seed=seed)

            try:
                plot_average_results(experiment_title, config, eval_csv_paths, train_csv_paths)
            except Exception as e:
                print("Error when trying to plot summary: " + str(e))
        else:
            train_csv_path = glob.glob(base_dir + str(config.random_seed) + "/*_train.csv")[0]
            eval_csv_path = glob.glob(base_dir + str(config.random_seed) + "/*_eval.csv")[0]
            attack_stats_csv_path = None
            try:
                attack_stats_csv_paths = glob.glob(base_dir + str(config.random_seed) + "/attack_stats_*.csv")
                attack_stats_csv_path = \
                list(filter(lambda x: "checkpoint" not in attack_stats_csv_paths, attack_stats_csv_paths))[0]
            except:
                pass
            train_csv_paths.append(train_csv_path)
            eval_csv_paths.append(eval_csv_path)
            plot_csv(config, eval_csv_path, train_csv_path, attack_stats_csv_path=attack_stats_csv_path,
                     random_seed=config.random_seed)
    else:
        if not config.run_many:
            run_experiment(args.configpath, 0, args.noconfig)
        else:
            train_csv_paths = []
            eval_csv_paths = []
            for seed in config.random_seeds:
                train_csv_path, eval_csv_path = run_experiment(args.configpath, seed, args.noconfig)
                train_csv_paths.append(train_csv_path)
                eval_csv_paths.append(eval_csv_path)
            try:
                plot_average_results(experiment_title, config, eval_csv_paths, train_csv_paths)
            except Exception as e:
                print("Error when trying to plot summary: " + str(e))



