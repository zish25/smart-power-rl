"""
Q-Learning Agent for Smart Electricity Load Management

Simple but powerful Q-Learning implementation with:
- Q-table for state-action values
- Epsilon-greedy exploration
- Experience replay concepts
"""

import numpy as np
from collections import defaultdict


class QLearningAgent:
    """
    Q-Learning agent for electricity load management

    State: Discretized [home_demand, hospital_demand, industry_demand, available_power]
    Actions: 3 discrete actions (0: home priority, 1: balanced, 2: hospital/industry priority)
    """

    def __init__(self, action_space=3, learning_rate=0.1, discount_factor=0.95,
                 epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01):
        """
        Initialize Q-Learning agent

        Args:
            action_space: Number of discrete actions (3)
            learning_rate: Alpha - how much new info overrides old (0.1)
            discount_factor: Gamma - importance of future rewards (0.95)
            epsilon: Initial exploration rate (1.0 = full exploration)
            epsilon_decay: Rate at which epsilon decreases per episode
            epsilon_min: Minimum epsilon value
        """
        self.action_space = action_space
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        # Q-table: dictionary mapping (discretized_state) -> {action: q_value}
        self.q_table = defaultdict(lambda: np.zeros(action_space))

        # Statistics
        self.training_steps = 0
        self.episode_rewards = []

    def _discretize_state(self, state):
        """
        Convert continuous state to discrete bins for Q-table

        State: [home_demand, hospital_demand, industry_demand, available_power]
        Each value is quantized into 10 bins
        """
        # Normalize state values to 0-10 range for discretization
        # Assumptions based on environment config
        limits = [40, 20, 60, 100]  # Max values for each dimension

        discretized = []
        for i, limit in enumerate(limits):
            bin_val = int((state[i] / limit) * 10)
            bin_val = min(10, max(0, bin_val))  # Clip to 0-10 range
            discretized.append(bin_val)

        return tuple(discretized)

    def select_action(self, state, training=True):
        """
        Select action using epsilon-greedy policy

        With probability epsilon: random action (exploration)
        With probability 1-epsilon: best action from Q-table (exploitation)

        Args:
            state: Current state from environment
            training: If True, use epsilon-greedy; if False, use greedy

        Returns:
            action: Integer 0, 1, or 2
        """
        discrete_state = self._discretize_state(state)

        if training and np.random.random() < self.epsilon:
            # Exploration: random action
            action = np.random.randint(0, self.action_space)
        else:
            # Exploitation: best action
            q_values = self.q_table[discrete_state]
            action = np.argmax(q_values)

        return action

    def update_q_table(self, state, action, reward, next_state, done):
        """
        Update Q-value using Q-Learning rule:

        Q(s,a) = Q(s,a) + α * [r + γ * max(Q(s',a')) - Q(s,a)]

        Where:
            α = learning rate (how much we update)
            r = reward received
            γ = discount factor (importance of future rewards)
            max(Q(s',a')) = best future action value

        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Resultant state
            done: Whether episode finished
        """
        discrete_state = self._discretize_state(state)
        discrete_next_state = self._discretize_state(next_state)

        # Current Q-value
        current_q = self.q_table[discrete_state][action]

        # Best future Q-value (or 0 if episode ended)
        if done:
            max_next_q = 0
        else:
            max_next_q = np.max(self.q_table[discrete_next_state])

        # Q-Learning update formula
        new_q = current_q + self.learning_rate * (reward + self.discount_factor * max_next_q - current_q)
        self.q_table[discrete_state][action] = new_q

        self.training_steps += 1

    def decay_epsilon(self):
        """Decay exploration rate after each episode"""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def record_episode_reward(self, total_reward):
        """Record total reward for the episode"""
        self.episode_rewards.append(total_reward)

    def get_statistics(self):
        """Get training statistics"""
        if len(self.episode_rewards) == 0:
            return {
                'episodes_completed': 0,
                'avg_reward_last_50': 0,
                'total_training_steps': self.training_steps,
                'current_epsilon': self.epsilon
            }

        return {
            'episodes_completed': len(self.episode_rewards),
            'avg_reward_last_50': np.mean(self.episode_rewards[-50:]),
            'avg_reward_all': np.mean(self.episode_rewards),
            'max_reward': np.max(self.episode_rewards),
            'total_training_steps': self.training_steps,
            'current_epsilon': self.epsilon
        }

    def get_policy_action(self, state):
        """Get best action without exploration (for testing)"""
        return self.select_action(state, training=False)

    def save_q_table(self, filepath):
        """Save Q-table to file"""
        import json
        # Convert q_table to JSON-serializable format
        q_dict = {str(k): v.tolist() for k, v in self.q_table.items()}
        with open(filepath, 'w') as f:
            json.dump(q_dict, f)
        print(f"Q-table saved to {filepath}")

    def load_q_table(self, filepath):
        """Load Q-table from file"""
        import json
        with open(filepath, 'r') as f:
            q_dict = json.load(f)
        self.q_table = defaultdict(lambda: np.zeros(self.action_space))
        for k, v in q_dict.items():
            # Convert string key back to tuple
            key = eval(k)
            self.q_table[key] = np.array(v)
        print(f"Q-table loaded from {filepath}")
