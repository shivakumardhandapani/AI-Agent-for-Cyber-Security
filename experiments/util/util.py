"""
Utility functions for experiments with the idsgame-env
"""
import io
import json
import jsonpickle
import logging
import time
import argparse
import os
from gym_idsgame.config.client_config import ClientConfig

def create_artefact_dirs(output_dir: str, random_seed : int) -> None:
    """
    Creates artefact directories if they do not already exist

    :param output_dir: the base directory
    :param random_seed: the random seed of the experiment
    :return: None
    """
    if not os.path.exists(output_dir + "/results/logs/" + str(random_seed) + "/"):
        os.makedirs(output_dir + "/results/logs/" + str(random_seed) + "/")
    if not os.path.exists(output_dir + "/results/plots/" + str(random_seed) + "/"):
        os.makedirs(output_dir + "/results/plots/" + str(random_seed) + "/")
    if not os.path.exists(output_dir + "/results/data/" + str(random_seed) + "/"):
        os.makedirs(output_dir + "/results/data/" + str(random_seed) + "/")
    if not os.path.exists(output_dir + "/results/hyperparameters/" + str(random_seed) + "/"):
        os.makedirs(output_dir + "/results/hyperparameters/" + str(random_seed) + "/")
    if not os.path.exists(output_dir + "/results/gifs/" + str(random_seed) + "/"):
        os.makedirs(output_dir + "/results/gifs/" + str(random_seed) + "/")
    if not os.path.exists(output_dir + "/results/tensorboard/" + str(random_seed) + "/"):
        os.makedirs(output_dir + "/results/tensorboard/" + str(random_seed) + "/")


def setup_logger(name: str, logdir: str, time_str = None):
    """
    Configures the logger for writing log-data of experiments

    :param name: name of the logger
    :param logdir: directory to save log files
    :param time_str: time string for file names
    :return: None
    """
    # create formatter
    formatter = logging.Formatter('%(asctime)s,%(message)s')
    # log to console
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    # log to file
    if time_str is None:
        time_str = str(time.time())
    fh = logging.FileHandler(logdir + "/" + time_str + "_" + name + ".log", mode="w")
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)

    # add the handlers to the logger
    logger.addHandler(fh)
    #logger.addHandler(ch)
    return logger

def write_config_file(config: ClientConfig, path: str) -> None:
    """
    Writes a config object to a config file

    :param config: the config to write
    :param path: the path to write the file
    :return: None
    """
    json_str = json.dumps(json.loads(jsonpickle.encode(config)), indent=4, sort_keys=True)
    with io.open(path, 'w', encoding='utf-8') as f:
        f.write(json_str)


def read_config(config_path) -> ClientConfig:
    """
    Reads configuration of the experiment from a json file

    :param config_path: the path to the configuration file
    :return: the configuration
    """
    with io.open(config_path, 'r', encoding='utf-8') as f:
        json_str = f.read()
    client_config: ClientConfig = jsonpickle.decode(json_str)
    return client_config


def parse_args(default_config_path):
    """
    Parses the commandline arguments with argparse

    :param default_config_path: default path to config file
    """
    parser = argparse.ArgumentParser(description='Parse flags to configure the json parsing')
    parser.add_argument("-cp", "--configpath", help="Path to configuration file",
                        default=default_config_path, type=str)
    parser.add_argument("-po", "--plotonly", help="Boolean parameter, if true, only plot",
                        action="store_true")
    parser.add_argument("-nc", "--noconfig", help="Boolean parameter, if true always override config",
                        action="store_true")
    args = parser.parse_args()
    return args