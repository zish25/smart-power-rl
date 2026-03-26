"""
Smart Electricity Load Management Environment
Custom OpenAI Gym-style environment for RL training

State: [home_demand, hospital_demand, industry_demand, available_power]
Actions: Allocation percentage for [homes, hospitals, industries]
Reward: Based on hospital priority, shortage penalties, and efficiency
"""

import numpy as np
from dataclasses import dataclass


@dataclass
class EnvironmentConfig:
    """Configuration for the environment"""
    max_power = 100.0  # Maximum available power (MW)
    home_demand_range = (10, 40)  # Typical home demand range
    hospital_demand_range = (5, 20)  # Critical hospital demand
    industry_demand_range = (20, 60)  # Industry demand range
    renewable_factor = 0.3  # Solar contribution (0-30% of max_power)


class LoadManagementEnv:
    """
    Custom environment for electricity load management

    Action space: 3 continuous values representing allocation percentage
    Observation space: 4 values [home_demand, hospital_demand, industry_demand, available_power]
    """

    def __init__(self, config=None):
        """Initialize the environment"""
        self.config = config or EnvironmentConfig()
        self.max_power = self.config.max_power

        # State variables
        self.home_demand = 0
        self.hospital_demand = 0
        self.industry_demand = 0
        self.available_power = self.max_power
        self.time_step = 0
        self.total_steps = 0

        # History for analysis
        self.history = {
            'rewards': [],
            'demands': [],
            'allocations': [],
            'shortages': [],
            'efficiency': []
        }

        # Initialize state
        self.reset()

    def reset(self):
        """Reset environment to initial state"""
        self.time_step = 0
        self.total_steps = 0
        self.home_demand = np.random.uniform(*self.config.home_demand_range)
        self.hospital_demand = np.random.uniform(*self.config.hospital_demand_range)
        self.industry_demand = np.random.uniform(*self.config.industry_demand_range)
        self.available_power = self._calculate_available_power()

        self.history = {
            'rewards': [],
            'demands': [],
            'allocations': [],
            'shortages': [],
            'efficiency': []
        }

        return self.get_state()

    def _calculate_available_power(self):
        """
        Calculate available power with renewable energy
        Solar varies: 0 during night, peaks during day
        """
        # Simulate solar variation (24-hour cycle)
        hour = (self.time_step % 24)

        # Solar availability: peak at hour 12 (noon)
        if 6 <= hour <= 18:
            solar_contribution = self.config.renewable_factor * self.max_power * np.sin((hour - 6) * np.pi / 12)
        else:
            solar_contribution = 0

        # Base power + renewable
        base_power = self.max_power * 0.85  # 85% base capacity
        return base_power + solar_contribution

    def _generate_emergency_spike(self):
        """
        Generate emergency spikes (hospitals may need emergency power)
        10% chance of spike per step
        """
        if np.random.random() < 0.1:
            spike = np.random.uniform(5, 15)  # Emergency spike
            return spike
        return 0

    def step(self, action):
        """
        Execute one step in the environment

        Action: [0, 2] for discrete actions representing allocation strategy
                Value 0: Prioritize homes
                Value 1: Balanced allocation
                Value 2: Prioritize hospitals and industry

        Returns: (state, reward, done, info)
        """
        self.time_step += 1
        self.total_steps += 1

        # Add dynamic demand with fluctuations
        home_noise = np.random.normal(0, 3)
        hospital_noise = np.random.normal(0, 2)
        industry_noise = np.random.normal(0, 5)
        emergency_spike = self._generate_emergency_spike()

        self.home_demand = np.clip(
            self.home_demand + home_noise,
            self.config.home_demand_range[0],
            self.config.home_demand_range[1]
        )

        self.hospital_demand = np.clip(
            self.hospital_demand + hospital_noise + emergency_spike,
            self.config.hospital_demand_range[0],
            self.config.hospital_demand_range[1]
        )

        self.industry_demand = np.clip(
            self.industry_demand + industry_noise,
            self.config.industry_demand_range[0],
            self.config.industry_demand_range[1]
        )

        # Update available power
        self.available_power = self._calculate_available_power()

        # Calculate allocations based on action
        home_alloc, hospital_alloc, industry_alloc = self._calculate_allocation(action)

        # Calculate shortages
        home_shortage = max(0, self.home_demand - home_alloc)
        hospital_shortage = max(0, self.hospital_demand - hospital_alloc)
        industry_shortage = max(0, self.industry_demand - industry_alloc)

        # Calculate wastage
        total_allocated = home_alloc + hospital_alloc + industry_alloc
        wastage = max(0, self.available_power - total_allocated)

        # Calculate reward
        reward = self._calculate_reward(
            home_shortage, hospital_shortage, industry_shortage,
            wastage, total_allocated
        )

        # Record history
        self.history['rewards'].append(reward)
        self.history['demands'].append([self.home_demand, self.hospital_demand, self.industry_demand])
        self.history['allocations'].append([home_alloc, hospital_alloc, industry_alloc])
        self.history['shortages'].append([home_shortage, hospital_shortage, industry_shortage])
        efficiency = total_allocated / self.available_power if self.available_power > 0 else 0
        self.history['efficiency'].append(efficiency)

        # Episode ends after 500 steps
        done = self.time_step >= 500

        info = {
            'home_shortage': home_shortage,
            'hospital_shortage': hospital_shortage,
            'industry_shortage': industry_shortage,
            'wastage': wastage,
            'efficiency': efficiency,
            'total_allocated': total_allocated
        }

        return self.get_state(), reward, done, info

    def _calculate_allocation(self, action):
        """
        Calculate power allocation based on action

        Action 0: Prioritize homes (50% homes, 35% hospitals, 15% industry)
        Action 1: Balanced (33% homes, 33% hospitals, 34% industry)
        Action 2: Prioritize hospitals & industry (20% homes, 40% hospitals, 40% industry)
        """
        # Ensure we never allocate more than available
        total_demand = self.home_demand + self.hospital_demand + self.industry_demand

        if action == 0:  # Home priority
            if total_demand > self.available_power:
                home_alloc = self.available_power * 0.50
                hospital_alloc = self.available_power * 0.35
                industry_alloc = self.available_power * 0.15
            else:
                home_alloc = self.home_demand
                hospital_alloc = self.hospital_demand
                industry_alloc = self.industry_demand

        elif action == 1:  # Balanced
            if total_demand > self.available_power:
                ratio = self.available_power / total_demand
                home_alloc = self.home_demand * ratio
                hospital_alloc = self.hospital_demand * ratio
                industry_alloc = self.industry_demand * ratio
            else:
                home_alloc = self.home_demand
                hospital_alloc = self.hospital_demand
                industry_alloc = self.industry_demand

        else:  # action == 2: Hospital & Industry priority
            if total_demand > self.available_power:
                hospital_alloc = min(self.hospital_demand, self.available_power * 0.40)
                industry_alloc = min(self.industry_demand, self.available_power * 0.40)
                home_alloc = min(self.home_demand, self.available_power - hospital_alloc - industry_alloc)
            else:
                home_alloc = self.home_demand
                hospital_alloc = self.hospital_demand
                industry_alloc = self.industry_demand

        return home_alloc, hospital_alloc, industry_alloc

    def _calculate_reward(self, home_shortage, hospital_shortage, industry_shortage, wastage, total_allocated):
        """
        Calculate reward based on:
        1. Hospital demand satisfaction (HIGHEST PRIORITY)
        2. Shortage penalties
        3. Efficiency (minimize wastage)
        4. Balance
        """
        # Hospital satisfaction is CRITICAL (40% weight)
        hospital_reward = -20 * hospital_shortage if hospital_shortage > 0 else 5

        # Home and industry shortages (30% weight each)
        home_reward = -5 * home_shortage if home_shortage > 0 else 2
        industry_reward = -3 * industry_shortage if industry_shortage > 0 else 1

        # Efficiency: penalize wastage (5% weight)
        wastage_penalty = -0.2 * wastage

        # Total reward
        total_reward = (
            hospital_reward * 0.4 +
            home_reward * 0.3 +
            industry_reward * 0.3 +
            wastage_penalty * 0.05
        )

        return total_reward

    def get_state(self):
        """Return current state as array [home_demand, hospital_demand, industry_demand, available_power]"""
        return np.array([
            self.home_demand,
            self.hospital_demand,
            self.industry_demand,
            self.available_power
        ], dtype=np.float32)

    def render(self):
        """Print current state (optional)"""
        print(f"Time: {self.time_step} | Home: {self.home_demand:.1f} | Hospital: {self.hospital_demand:.1f} | "
              f"Industry: {self.industry_demand:.1f} | Available: {self.available_power:.1f}")
