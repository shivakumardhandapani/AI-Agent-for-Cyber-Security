# Experiment `random_defense-v0`_`dqn`

This is an experiment in the `random_defense-v0` environment. 
An environment where the defender is following a random defense policy. 
This experiment trains an attacker agent using the DQN algorithm to act optimally in the given
environment and defeat the random defender.

The network configuration of the environment is as follows:

- `num_layers=1` (number of layers between the start and end nodes)
- `num_servers_per_layer=1`
- `num_attack_types=10`
- `max_value=9`  

<p align="center">
<img src="docs/env.png" width="600">
</p>

The starting state for each node in the environment is initialized as follows (with some randomness for where the vulnerabilities are placed).

- `defense_val=2`
- `attack_val=0`
- `num_vulnerabilities_per_node=1` (which type of defense at the node that is vulnerable is selected randomly when the environment is initialized)
- `det_val=2`
- `vulnerability_val=0` 
- `num_vulnerabilities_per_layer=1`

The environment has sparse rewards (+1,-1 rewards are given at the terminal state of each episode).
 The environment is partially observed (attacker can only see attack attributes of neighboring nodes, defender can only see defense attributes)

## Environment 

- Env: `random_defense-v0`

## Algorithm

- DQN with linear exploration annealing 
 
## Instructions 

To configure the experiment use the `config.json` file. Alternatively, 
it is also possible to delete the config file and edit the configuration directly in
`run.py` (this will cause the configuration to be overridden on the next run). 

Example configuration in `config.json`:

```json
{
    "attacker_type": 6,
    "defender_type": 1,
    "env_name": "idsgame-random_defense-v0",
    "idsgame_config": null,
    "initial_state_path": null,
    "logger": null,
    "mode": 0,
    "output_dir": "/home/kim/storage/workspace/gym-idsgame/experiments/training/v0/random_defense/dqn",
    "py/object": "gym_idsgame.config.client_config.ClientConfig",
    "q_agent_config": {
        "alpha": 0.0001,
        "attacker": true,
        "checkpoint_freq": 100000,
        "defender": false,
        "dqn_config": {
            "batch_size": 32,
            "gpu": true,
            "hidden_activation": "ReLU",
            "hidden_dim": 64,
            "input_dim": 33,
            "loss_fn": "Huber",
            "lr_decay_rate": 0.999,
            "lr_exp_decay": true,
            "num_hidden_layers": 1,
            "optimizer": "Adam",
            "output_dim": 30,
            "py/object": "gym_idsgame.agents.q_learning.dqn.dqn_config.DQNConfig",
            "replay_memory_size": 1000,
            "replay_start_size": 100,
            "target_network_update_freq": 100,
            "tensorboard": true,
            "tensorboard_dir": "/home/kim/storage/workspace/gym-idsgame/experiments/training/v0/random_defense/dqn/tensorboard"
        },
        "epsilon": 1,
        "epsilon_decay": 0.999,
        "eval_episodes": 100,
        "eval_frequency": 1000,
        "eval_log_frequency": 1,
        "eval_render": false,
        "eval_sleep": 0.9,
        "gamma": 0.9,
        "gif_dir": "/home/kim/storage/workspace/gym-idsgame/experiments/training/v0/random_defense/dqn/gifs",
        "gifs": true,
        "load_path": null,
        "logger": null,
        "min_epsilon": 0.01,
        "num_episodes": 5000,
        "py/object": "gym_idsgame.agents.q_learning.q_agent_config.QAgentConfig",
        "render": false,
        "save_dir": "/home/kim/storage/workspace/gym-idsgame/experiments/training/v0/random_defense/dqn/data",
        "train_log_frequency": 1,
        "video": true,
        "video_dir": "/home/kim/storage/workspace/gym-idsgame/experiments/training/v0/random_defense/dqn/videos",
        "video_fps": 5,
        "video_frequency": 101
    },
    "simulation_config": null,
    "title": "TrainingDQNAgent vs RandomDefender"
}
```

Example configuration in `run.py`:

```python
dqn_config = DQNConfig(input_dim=33, output_dim=30, hidden_dim=64, replay_memory_size=1000,
                       num_hidden_layers=1,
                       replay_start_size=100, batch_size=32, target_network_update_freq=100,
                       gpu=True, tensorboard=True, tensorboard_dir=default_output_dir() + "/tensorboard",
                       loss_fn="Huber", optimizer="Adam", lr_exp_decay=True, lr_decay_rate=0.999)
q_agent_config = QAgentConfig(gamma=0.9, alpha=0.0001, epsilon=1, render=False, eval_sleep=0.9,
                              min_epsilon=0.01, eval_episodes=100, train_log_frequency=1,
                              epsilon_decay=0.999, video=True, eval_log_frequency=1,
                              video_fps=5, video_dir=default_output_dir() + "/videos", num_episodes=5000,
                              eval_render=False, gifs=True, gif_dir=default_output_dir() + "/gifs",
                              eval_frequency=1000, attacker=True, defender=False, video_frequency=101,
                              save_dir=default_output_dir() + "/data", dqn_config=dqn_config)
env_name = "idsgame-random_defense-v0"
client_config = ClientConfig(env_name=env_name, attacker_type=AgentType.DQN_AGENT.value,
                             mode=RunnerMode.TRAIN_ATTACKER.value,
                             q_agent_config=q_agent_config, output_dir=default_output_dir(),
                             title="TrainingDQNAgent vs RandomDefender")
```

After the experiment has finished, the results are written to the following sub-directories:

- **/data**: CSV file with metrics per episode for train and eval, e.g. `avg_episode_rewards`, `avg_episode_steps`, etc.
- **/gifs**: If the gif configuration-flag is set to true, the experiment will render the game during evaluation and save gif files to this directory. You can control the frequency of evaluation with the configuration parameter `eval_frequency` and the frequency of video/gif recording during evaluation with the parameter `video_frequency`
- **/hyperparameters**: CSV file with hyperparameters for the experiment.
- **/logs**: Log files from the experiment
- **/plots**: Basic plots of the results
- **/videos**: If the video configuration-flag is set to true, the experiment will render the game during evaluation and save video files to this directory. You can control the frequency of evaluation with the configuration parameter `eval_frequency` and the frequency of video/gif recording during evaluation with the parameter `video_frequency`
  

## Example Results

### Hack probability

#### Train

<p align="center">
<img src="docs/hack_probability_train.png" width="800">
</p>

#### Eval

<p align="center">
<img src="docs/hack_probability_eval.png" width="800">
</p>

### Episode lengths

#### Train
<p align="center">
<img src="docs/avg_episode_lengths_train.png" width="800">
</p>

#### Eval

<p align="center">
<img src="docs/avg_episode_lengths_eval.png" width="800">
</p>

### Exploration Rate

<p align="center">
<img src="docs/epsilon_train.png" width="800">
</p>

### Cumulative Rewards

#### Attacker (Train)
<p align="center">
<img src="docs/attacker_cumulative_reward_train.png" width="800">
</p>

#### Defender (Train)
<p align="center">
<img src="docs/defender_cumulative_reward_train.png" width="800">
</p>

### Loss

#### Attacker (Train)
<p align="center">
<img src="docs/avg_episode_loss_attacker_train.png" width="800">
</p>

### Learning Rate Decay

#### Attacker (Train)
<p align="center">
<img src="docs/lr_train.png" width="800">
</p>

### Policy Inspection

#### Evaluation after 0 Training Episodes

<p align="center">
<img src="docs/episode_0.gif" width="600">
</p> 

#### Evaluation after 1000 Training Episodes

<p align="center">
<img src="docs/episode_1000.gif" width="600">
</p>

#### Evaluation after 5000 Training Episodes

<p align="center">
<img src="docs/episode_5000.gif" width="600">
</p>    

## Commands

Below is a list of commands for running the experiment

### Run

**Option 1**:
```bash
./run.sh
```

**Option 2**:
```bash
make all
```

**Option 3**:
```bash
python run.py
```

### Run Server (Without Display)

**Option 1**:
```bash
./run_server.sh
```

**Option 2**:
```bash
make run_server
```

### Clean

```bash
make clean
```

### Tensorboard

```
tensorboard --logdir=tensorboard
```

or: 

```
make tensorboard
```