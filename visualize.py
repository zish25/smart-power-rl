"""
Visualization module for Smart Electricity Load Management

Creates plots showing:
- Reward improvement over episodes
- Efficiency improvement over time
- Demand vs Supply during simulation
- Hospital shortage reduction
"""

import matplotlib.pyplot as plt
import numpy as np


def plot_training_results(training_history):
    """
    Plot training results in a 2x2 subplot layout

    Args:
        training_history: Dictionary with training metrics
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Smart Electricity Load Management - Training Results', fontsize=16, fontweight='bold')

    # Plot 1: Episode Rewards
    ax1 = axes[0, 0]
    episodes = range(len(training_history['episode_rewards']))
    ax1.plot(episodes, training_history['episode_rewards'], linewidth=1, alpha=0.6, label='Episode Reward')

    # Add moving average
    window = 50
    if len(training_history['episode_rewards']) >= window:
        moving_avg = np.convolve(training_history['episode_rewards'], np.ones(window)/window, mode='valid')
        ax1.plot(range(window-1, len(training_history['episode_rewards'])), moving_avg,
                linewidth=2, label=f'Moving Avg ({window} episodes)', color='red')

    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Total Reward')
    ax1.set_title('Agent Reward Improvement Over Time')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Efficiency Over Episodes
    ax2 = axes[0, 1]
    ax2.plot(episodes, training_history['episode_avg_efficiency'], linewidth=1, alpha=0.6, label='Efficiency')

    if len(training_history['episode_avg_efficiency']) >= window:
        moving_avg_eff = np.convolve(training_history['episode_avg_efficiency'],
                                     np.ones(window)/window, mode='valid')
        ax2.plot(range(window-1, len(training_history['episode_avg_efficiency'])), moving_avg_eff,
                linewidth=2, label=f'Moving Avg', color='red')

    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Power Efficiency')
    ax2.set_title('Grid Efficiency Improvement (Power Used / Available)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 1.1])

    # Plot 3: Hospital Shortage Reduction
    ax3 = axes[1, 0]
    ax3.plot(episodes, training_history['episode_hospital_shortage'], linewidth=1, alpha=0.6, label='Hospital Shortage')

    if len(training_history['episode_hospital_shortage']) >= window:
        moving_avg_hosp = np.convolve(training_history['episode_hospital_shortage'],
                                      np.ones(window)/window, mode='valid')
        ax3.plot(range(window-1, len(training_history['episode_hospital_shortage'])), moving_avg_hosp,
                linewidth=2, label=f'Moving Avg', color='red')

    ax3.set_xlabel('Episode')
    ax3.set_ylabel('Average Shortage (MW)')
    ax3.set_title('Hospital Power Shortage Over Time (Should Decrease)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Plot 4: Exploration Rate (Epsilon)
    ax4 = axes[1, 1]
    ax4.plot(episodes, training_history['episode_epsilon'], linewidth=2, color='purple')
    ax4.set_xlabel('Episode')
    ax4.set_ylabel('Epsilon (e)')
    ax4.set_title('Exploration Rate Decay (epsilon-greedy)')
    ax4.set_yscale('log')
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_simulation(env_history, title="Electricity Load Management Simulation"):
    """
    Plot a single simulation showing demand vs allocation

    Args:
        env_history: History from an environment run
        title: Plot title
    """
    demands = np.array(env_history['demands'])
    allocations = np.array(env_history['allocations'])
    efficiency = np.array(env_history['efficiency'])

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(title, fontsize=16, fontweight='bold')

    # Time steps
    time_steps = range(len(demands))

    # Plot 1: Home demand vs allocation
    ax1 = axes[0, 0]
    ax1.plot(time_steps, demands[:, 0], label='Demand', linewidth=2, linestyle='--')
    ax1.plot(time_steps, allocations[:, 0], label='Allocated', linewidth=2)
    ax1.fill_between(time_steps, allocations[:, 0], demands[:, 0], alpha=0.3, color='red', label='Shortage')
    ax1.set_ylabel('Power (MW)')
    ax1.set_title('HOME SECTOR')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Hospital demand vs allocation (MOST CRITICAL)
    ax2 = axes[0, 1]
    ax2.plot(time_steps, demands[:, 1], label='Demand', linewidth=2, linestyle='--', color='orange')
    ax2.plot(time_steps, allocations[:, 1], label='Allocated', linewidth=2, color='red')
    ax2.fill_between(time_steps, allocations[:, 1], demands[:, 1], alpha=0.3, color='red', label='Shortage')
    ax2.set_ylabel('Power (MW)')
    ax2.set_title('HOSPITAL SECTOR (Critical Priority)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Industry demand vs allocation
    ax3 = axes[1, 0]
    ax3.plot(time_steps, demands[:, 2], label='Demand', linewidth=2, linestyle='--', color='green')
    ax3.plot(time_steps, allocations[:, 2], label='Allocated', linewidth=2, color='darkgreen')
    ax3.fill_between(time_steps, allocations[:, 2], demands[:, 2], alpha=0.3, color='red', label='Shortage')
    ax3.set_xlabel('Time Step')
    ax3.set_ylabel('Power (MW)')
    ax3.set_title('INDUSTRY SECTOR')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Plot 4: Overall efficiency
    ax4 = axes[1, 1]
    ax4.plot(time_steps, efficiency, linewidth=2, color='purple')
    ax4.axhline(y=0.8, color='g', linestyle='--', label='Target (80%)')
    ax4.fill_between(time_steps, 0, efficiency, alpha=0.3, color='purple')
    ax4.set_xlabel('Time Step')
    ax4.set_ylabel('Efficiency Ratio')
    ax4.set_ylim([0, 1.1])
    ax4.set_title('Grid Efficiency (Power Used / Available)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_comparison(before_history, after_history):
    """
    Compare performance before and after training

    Args:
        before_history: History from untrained agent
        after_history: History from trained agent
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Performance Comparison: Before vs After Training', fontsize=16, fontweight='bold')

    sectors = ['Homes', 'Hospitals', 'Industry']

    # Calculate metrics for both
    before_shortages = np.array(before_history['shortages']).mean(axis=0)
    after_shortages = np.array(after_history['shortages']).mean(axis=0)

    before_efficiency = np.mean(before_history['efficiency'])
    after_efficiency = np.mean(after_history['efficiency'])

    before_rewards = np.mean(before_history['rewards'])
    after_rewards = np.mean(after_history['rewards'])

    # Plot 1: Average Shortage Comparison
    ax1 = axes[0, 0]
    x = np.arange(len(sectors))
    width = 0.35
    ax1.bar(x - width/2, before_shortages, width, label='Before Training', alpha=0.8)
    ax1.bar(x + width/2, after_shortages, width, label='After Training', alpha=0.8)
    ax1.set_ylabel('Average Shortage (MW)')
    ax1.set_title('Power Shortage by Sector')
    ax1.set_xticks(x)
    ax1.set_xticklabels(sectors)
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')

    # Plot 2: Efficiency Comparison
    ax2 = axes[0, 1]
    categories = ['Before', 'After']
    efficiencies = [before_efficiency, after_efficiency]
    colors = ['#ff7f0e', '#2ca02c']
    bars = ax2.bar(categories, efficiencies, color=colors, alpha=0.8, width=0.5)
    ax2.set_ylabel('Efficiency Ratio')
    ax2.set_title('Grid Efficiency Improvement')
    ax2.set_ylim([0, 1])
    for bar, eff in zip(bars, efficiencies):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{eff:.2%}', ha='center', va='bottom', fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')

    # Plot 3: Average Reward
    ax3 = axes[1, 0]
    rewards = [before_rewards, after_rewards]
    bars = ax3.bar(categories, rewards, color=colors, alpha=0.8, width=0.5)
    ax3.set_ylabel('Average Reward')
    ax3.set_title('Agent Reward Improvement')
    for bar, reward in zip(bars, rewards):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{reward:.2f}', ha='center', va='bottom', fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')

    # Plot 4: Text summary
    ax4 = axes[1, 1]
    ax4.axis('off')

    summary_text = f"""
    KEY IMPROVEMENTS:

    Hospital Shortage: {before_shortages[1]:.2f} → {after_shortages[1]:.2f} MW
    Reduction: {(before_shortages[1] - after_shortages[1])/before_shortages[1]*100:.1f}%

    Grid Efficiency: {before_efficiency:.2%} → {after_efficiency:.2%}
    Improvement: {(after_efficiency - before_efficiency)/before_efficiency*100:.1f}%

    Average Reward: {before_rewards:.2f} → {after_rewards:.2f}
    Improvement: {((after_rewards - before_rewards)/abs(before_rewards)*100):.1f}%
    """

    ax4.text(0.1, 0.5, summary_text, fontsize=12, verticalalignment='center',
            family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    return fig


def save_plots(fig, filename):
    """Save figure to file"""
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {filename}")
