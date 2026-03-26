"""
Main Script - Smart Electricity Load Management using RL

Run this script to:
1. Train a Q-Learning agent
2. Evaluate the trained agent
3. Generate visualizations
4. Display results
"""

import sys
import numpy as np
from environment import LoadManagementEnv, EnvironmentConfig
from agent import QLearningAgent
from train import train_agent, evaluate_agent
from visualize import plot_training_results, plot_simulation, plot_comparison, save_plots
import matplotlib.pyplot as plt


def run_untrained_agent_simulation(num_steps=500):
    """
    Run a simulation with random actions to show baseline performance

    Args:
        num_steps: Number of steps to simulate

    Returns:
        history from environment
    """
    print("\n" + "=" * 80)
    print("SIMULATING UNTRAINED (RANDOM) AGENT")
    print("=" * 80)

    env = LoadManagementEnv()
    state = env.reset()

    for _ in range(num_steps):
        # Random action
        action = np.random.randint(0, 3)
        state, reward, done, info = env.step(action)
        if done:
            break

    print(f"Completed {len(env.history['rewards'])} steps with random agent")
    print(f"Average Reward: {np.mean(env.history['rewards']):.2f}")
    print(f"Average Efficiency: {np.mean(env.history['efficiency']):.4f}")

    return env.history


def print_state_action_example(agent):
    """
    Print example of what agent learned (state-action pairs)

    Args:
        agent: Trained agent
    """
    print("\n" + "=" * 80)
    print("LEARNED POLICY EXAMPLES")
    print("=" * 80)
    print("\nAction meanings:")
    print("  Action 0: Prioritize Homes (50%, 35% hospitals, 15% industry)")
    print("  Action 1: Balanced Allocation (33% each sector)")
    print("  Action 2: Prioritize Hospitals & Industry (20%, 40%, 40%)")

    print("\nExample learned states (showing best action):")
    print("\nState Format: [Home Demand (bin), Hospital Demand (bin), Industry Demand (bin), Available Power (bin)]")
    print("-" * 100)

    # Show some learned states
    action_names = {0: "HOME PRIORITY", 1: "BALANCED", 2: "HOSPITAL/INDUSTRY PRIORITY"}
    example_states = [
        (10, 10, 2, 10),  # High demand for homes/hospitals, low industry, high power
        (2, 10, 2, 5),    # Low homes, high hospitals, low industry, medium power
        (5, 5, 10, 5),    # Balanced demand, medium power
        (10, 10, 10, 3),  # High demand all, low available power
    ]

    for state_tuple in example_states:
        q_values = agent.q_table[state_tuple]
        best_action = np.argmax(q_values)
        print(f"State {state_tuple} -> Best Action: {best_action} ({action_names[best_action]})")
        print(f"  Q-values: [Home: {q_values[0]:.2f}, Balanced: {q_values[1]:.2f}, Hospital: {q_values[2]:.2f}]")


def main():
    """Main execution function"""

    print("\n" + "#" * 80)
    print("#" + " " * 78 + "#")
    print("#" + "  SMART ELECTRICITY LOAD MANAGEMENT USING REINFORCEMENT LEARNING".center(78) + "#")
    print("#" + "  India-focused Power Distribution Optimization".center(78) + "#")
    print("#" + " " * 78 + "#")
    print("#" * 80)

    # Step 1: Get baseline performance with random agent
    print("\n[STEP 1/5] Simulating untrained agent...")
    untrained_history = run_untrained_agent_simulation()

    # Step 2: Train Q-Learning agent
    print("\n[STEP 2/5] Training Q-Learning agent...")
    agent, env, training_history = train_agent(num_episodes=500, verbose=True)

    # Step 3: Evaluate trained agent
    print("\n[STEP 3/5] Evaluating trained agent...")
    eval_history = evaluate_agent(agent, env, num_episodes=10, verbose=False)

    # Get final simulation for visualization
    print("\n[STEP 4/5] Running final simulation for visualization...")
    state = env.reset()
    done = False
    step_count = 0
    while not done and step_count < 500:
        action = agent.get_policy_action(state)
        state, reward, done, info = env.step(action)
        step_count += 1

    final_simulation_history = env.history

    # Step 5: Generate visualizations
    print("\n[STEP 5/5] Generating visualizations...")

    # Training results
    fig1 = plot_training_results(training_history)
    save_plots(fig1, "01_training_results.png")

    # Final simulation
    fig2 = plot_simulation(final_simulation_history, "Final Simulation with Trained Agent")
    save_plots(fig2, "02_final_simulation.png")

    # Comparison
    fig3 = plot_comparison(untrained_history, final_simulation_history)
    save_plots(fig3, "03_performance_comparison.png")

    # Print learned policy
    print_state_action_example(agent)

    # Final summary
    print("\n" + "=" * 80)
    print("FINAL RESULTS & PERFORMANCE METRICS")
    print("=" * 80)

    untrained_efficiency = np.mean(untrained_history['efficiency'])
    trained_efficiency = np.mean(final_simulation_history['efficiency'])

    untrained_reward = np.mean(untrained_history['rewards'])
    trained_reward = np.mean(final_simulation_history['rewards'])

    hospital_shortage_untrained = np.mean([s[1] for s in untrained_history['shortages']])
    hospital_shortage_trained = np.mean([s[1] for s in final_simulation_history['shortages']])

    print(f"\n{'Metric':<40} {'Untrained':<20} {'Trained':<20} {'Improvement':<20}")
    print("-" * 100)
    print(f"{'Grid Efficiency':<40} {untrained_efficiency:<20.2%} {trained_efficiency:<20.2%} {((trained_efficiency/untrained_efficiency - 1)*100):<20.1f}%")
    print(f"{'Average Reward per Step':<40} {untrained_reward:<20.2f} {trained_reward:<20.2f} {((trained_reward - untrained_reward)/abs(untrained_reward)*100):<20.1f}%")
    print(f"{'Hospital Shortage (MW)':<40} {hospital_shortage_untrained:<20.2f} {hospital_shortage_trained:<20.2f} {((hospital_shortage_untrained - hospital_shortage_trained)/hospital_shortage_untrained*100):<20.1f}%")

    print("\n" + "=" * 80)
    print("OUTPUT FILES GENERATED:")
    print("=" * 80)
    print("[OK] 01_training_results.png - Training progress and learning curves")
    print("[OK] 02_final_simulation.png - Final simulation with trained agent")
    print("[OK] 03_performance_comparison.png - Before vs After comparison")
    print("[OK] q_table.json - Trained Q-table (can be reused)")

    print("\n" + "=" * 80)
    print("PROJECT COMPLETE!")
    print("=" * 80)

    # Display plots
    plt.show()


if __name__ == "__main__":
    main()
