"""
Training Pipeline for Smart Electricity Load Management

Trains the Q-Learning agent on the environment and collects metrics
"""

from environment import LoadManagementEnv, EnvironmentConfig
from agent import QLearningAgent
import numpy as np


def train_agent(num_episodes=500, verbose=True):
    """
    Train the Q-Learning agent

    Args:
        num_episodes: Number of episodes to train
        verbose: Print progress during training

    Returns:
        agent: Trained agent
        env: Environment
        training_history: Metrics from training
    """
    # Initialize environment and agent
    env_config = EnvironmentConfig()
    env = LoadManagementEnv(config=env_config)
    agent = QLearningAgent(
        action_space=3,
        learning_rate=0.1,
        discount_factor=0.95,
        epsilon=1.0,
        epsilon_decay=0.995,
        epsilon_min=0.01
    )

    # Training history
    training_history = {
        'episode_rewards': [],
        'episode_avg_efficiency': [],
        'episode_hospital_shortage': [],
        'episode_home_shortage': [],
        'episode_industry_shortage': [],
        'episode_epsilon': []
    }

    print("=" * 80)
    print("STARTING Q-LEARNING TRAINING")
    print("=" * 80)
    print(f"Environment: Smart Electricity Load Management")
    print(f"Agent: Q-Learning with epsilon-greedy exploration")
    print(f"Episodes: {num_episodes}")
    print(f"Max steps per episode: 500")
    print("=" * 80)

    # Training loop
    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        episode_hospital_shortage = 0
        episode_efficiency = 0
        episode_steps = 0

        # Episode loop
        done = False
        step_count = 0
        while not done:
            # Agent selects action
            action = agent.select_action(state, training=True)

            # Environment step
            next_state, reward, done, info = env.step(action)

            # Update Q-table
            agent.update_q_table(state, action, reward, next_state, done)

            # Accumulate metrics
            episode_reward += reward
            episode_hospital_shortage += info['hospital_shortage']
            episode_efficiency += info['efficiency']
            episode_steps += 1

            state = next_state
            step_count += 1

        # Decay epsilon
        agent.decay_epsilon()

        # Record episode metrics
        agent.record_episode_reward(episode_reward)
        training_history['episode_rewards'].append(episode_reward)
        training_history['episode_avg_efficiency'].append(episode_efficiency / episode_steps)
        training_history['episode_hospital_shortage'].append(episode_hospital_shortage / episode_steps)
        training_history['episode_epsilon'].append(agent.epsilon)

        # Print progress
        if verbose and (episode + 1) % 50 == 0:
            stats = agent.get_statistics()
            print(f"\nEpisode {episode + 1}/{num_episodes}")
            print(f"  Total Reward: {episode_reward:.2f}")
            print(f"  Avg Efficiency: {stats['avg_reward_last_50']:.2f}")
            print(f"  Avg Hospital Shortage: {training_history['episode_hospital_shortage'][-1]:.2f}")
            print(f"  Epsilon (exploration rate): {agent.epsilon:.4f}")
            print(f"  Q-table states: {len(agent.q_table)}")

    print("\n" + "=" * 80)
    print("TRAINING COMPLETED")
    print("=" * 80)

    stats = agent.get_statistics()
    print(f"Total Episodes: {stats['episodes_completed']}")
    print(f"Total Training Steps: {stats['total_training_steps']}")
    print(f"Average Reward (Last 50 episodes): {stats['avg_reward_last_50']:.2f}")
    print(f"Best Episode Reward: {stats['max_reward']:.2f}")
    print(f"Final Epsilon: {stats['current_epsilon']:.4f}")

    return agent, env, training_history


def evaluate_agent(agent, env, num_episodes=10, verbose=True):
    """
    Evaluate trained agent without exploration

    Args:
        agent: Trained agent
        env: Environment
        num_episodes: Number of evaluation episodes
        verbose: Print details

    Returns:
        eval_history: Evaluation metrics
    """
    eval_history = {
        'episode_rewards': [],
        'episode_efficiency': [],
        'episode_hospital_shortage': [],
        'episode_home_shortage': [],
        'episode_industry_shortage': []
    }

    print("\n" + "=" * 80)
    print("EVALUATING TRAINED AGENT")
    print("=" * 80)

    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        episode_efficiency = 0
        episode_hospital_shortage = 0
        episode_home_shortage = 0
        episode_industry_shortage = 0
        episode_steps = 0

        done = False
        while not done:
            # Use best action (no exploration)
            action = agent.get_policy_action(state)

            # Environment step
            next_state, reward, done, info = env.step(action)

            episode_reward += reward
            episode_efficiency += info['efficiency']
            episode_hospital_shortage += info['hospital_shortage']
            episode_home_shortage += info['home_shortage']
            episode_industry_shortage += info['industry_shortage']
            episode_steps += 1

            state = next_state

        # Record metrics
        eval_history['episode_rewards'].append(episode_reward)
        eval_history['episode_efficiency'].append(episode_efficiency / episode_steps)
        eval_history['episode_hospital_shortage'].append(episode_hospital_shortage / episode_steps)
        eval_history['episode_home_shortage'].append(episode_home_shortage / episode_steps)
        eval_history['episode_industry_shortage'].append(episode_industry_shortage / episode_steps)

        if verbose:
            print(f"\nEvaluation Episode {episode + 1}/{num_episodes}")
            print(f"  Total Reward: {episode_reward:.2f}")
            print(f"  Avg Efficiency: {episode_efficiency / episode_steps:.4f}")
            print(f"  Avg Hospital Shortage: {episode_hospital_shortage / episode_steps:.2f}")
            print(f"  Avg Home Shortage: {episode_home_shortage / episode_steps:.2f}")
            print(f"  Avg Industry Shortage: {episode_industry_shortage / episode_steps:.2f}")

    print("\n" + "=" * 80)
    print("EVALUATION SUMMARY")
    print("=" * 80)
    avg_reward = np.mean(eval_history['episode_rewards'])
    avg_efficiency = np.mean(eval_history['episode_efficiency'])
    avg_hospital_shortage = np.mean(eval_history['episode_hospital_shortage'])

    print(f"Average Reward: {avg_reward:.2f}")
    print(f"Average Efficiency: {avg_efficiency:.4f}")
    print(f"Average Hospital Shortage: {avg_hospital_shortage:.2f}")

    return eval_history


if __name__ == "__main__":
    # Train agent
    agent, env, training_history = train_agent(num_episodes=500, verbose=True)

    # Evaluate agent
    eval_history = evaluate_agent(agent, env, num_episodes=10, verbose=True)

    # Save agent
    agent.save_q_table("q_table.json")

    print("\nTraining and evaluation completed!")
