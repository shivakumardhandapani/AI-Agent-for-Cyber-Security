In recent years, reinforcement learning (RL) has emerged as a powerful tool for simulating and responding to cyber attacks in dynamic environments. This project explores the use of RL algorithms for both offensive and defensive strategies 
within a simulated intrusion detection system called gymidsgame, an Abstract Cyber Security Simulation and Markov Game for OpenAI Gym. gym-idsame models a grid-world where an attacker attempts to exploit system nodes while a defender
secures them. Four RL algorithms—Q-Learning, SARSA (State Action Rewards State’ Action’), Deep Q-Network (DQN), and Double Deep Q-Network (DDQN) are implemented and compared . Each agent interacts with the environment over multiple
episodes and receives scalar rewards based on outcomes such as successful intrusions or defense actions. The attacker and defender rewards, epsilon decay, and hack probability across training runs were documented.

Environment: 
  gym-idsgame is a reinforcement learning environment designed for simulating attack and defense operations in an abstract network intrusion game. Offers multiple registered environments for different scenarios, such as minimal defense, maximal attack, random attack, and two-agent simulations.Supports training with algorithms like Tabular Q-learning, Neural-fitted Q-learning (DQN), REINFORCE, 
  Actor-Critic REINFORCE, and PPO. At each timestep, both the attacker and defender agents take actions simultaneously. The outcome of their interactions affects the environment state and determines the rewards
  assigned to each agent. The state representation—also referred to as the observation space—is a flattened numerical vector encoding the status of each node in the network, including their vulnerability, compromise state, and any applied defense
  mechanisms. In our setup, the state space has 33 dimensions, and the attacker’s action space consists of 30 discrete actions, each corresponding to a specific attack on a particular node.

Algorithms being analysed:

  Q-Learning: Q-Learning is an off-policy learning method. It updates the Q-value for a certain action based on the obtained reward from the next state and the maximum reward from the possible states after that. It is off-policy because it
  uses an ϵ-greedy strategy for the first step and a greedy action selection strategy for the second step.
  
  DQN: DQN algorithm extends on traditional Qlearning algorithm by integrating neural networks to estimate the Q-function, enabling agents to operate in high-dimensional and continuous state spaces. In contrast to tabular Q-learning,
  where state-action values are stored explicitly, DQN approximates the Q-values using a deep neural network that takes the current state as input and outputs Q-values for all possible
  actions.
  
  Double DQN: The DDQN algorithm is an enhancement over the traditional DQN, specifically designed to mitigate the problem of Q-value overestimation during training. In standard DQN, both the action selection and action evaluation are
  performed using the same Q-network. This can lead to overly optimistic value estimates, especially in environments with high variance or sparse rewards, such as network security
  simulations.
  DDQN introduces a target Q-network that is a periodically updated copy of the main Q-network. During training, the main network is used to select the best action from the next state s′, and the target network is used to evaluate the Qvalue
  of that action. This separation helps stabilize learning and improve accuracy in estimating future rewards.
  
  SARSA: SARSA is an on-policy RL algorithm that updates the action-value function Q(s, a) based on the current state, action, reward, next state, and next action. Unlike Q-learning, which estimates the optimal policy regardless
  of the agent’s behavior, SARSA evaluates and improves the policy that is actually being followed, making it inherently more conservative and stable in certain environments.

Conclusion: 
  In summary, Q-Learning, SARSA, DQN and DDQN have been applied to the gym-idsgame environment to simulate how attacker and defender performed under the defensive strategies. Overall, all the algorithms being explored performs
  well and effectively learns a defensive strategy to reduce hack probability in Cybersecurity world.
  Different agents performs slightly differently based on observation and results analysis, SARSA is safe and cautious, good for environments with penalties for wrong actions . QLearning
  can be riskier but faster, potentially learning aggressive defense faster but possibly unstable. DDQN has relatively best performance considering best overall reward and hack
  reduction, especially in complex or large environments. But it requires more time, GPU and tuning.
