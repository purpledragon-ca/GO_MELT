"""
Reward Computation for GO-MELT RL Environment

This module provides reward computation functions for reinforcement learning
in the GO-MELT simulation environment. It includes:
- Temperature-based reward: based on error from target temperature
- Action penalty: optional penalty to encourage smaller actions and smoother control
"""

import numpy as np


def compute_reward_temperature_error(
    temperature: float,
    target_temperature: float = 3000.0, 
) -> float:

    error = target_temperature - temperature
    normalized_error = error / target_temperature
    temperature_error = (normalized_error ** 2)
    return float(temperature_error)


def compute_reward_temperature_error_tanh(
    temperature: float,
    target_temperature: float = 3000.0,
    std: float = 10.0
) -> float:
    temp_error = np.abs(target_temperature - temperature)
    temp_error_tanh = 1.0 - np.tanh(temp_error / std)
    return float(temp_error_tanh)


def compute_action_penalty(
    action: float,
) -> float:
    return float(action)


def compute_total_reward(
    temperature: float,
    target_temperature: float = 3000.0,
    action: float = 0.0
) -> float:
    TEMPERATURE_ERROR_COEF = -1.0
    TEMPERATURE_ERROR_TANH_COEF = 1.0
    ACTION_PENALTY_COEF = -0.01
    
    temperature_error = compute_reward_temperature_error(
        temperature=temperature,
        target_temperature=target_temperature,
    )

    temp_error_tanh = compute_reward_temperature_error_tanh(
        temperature=temperature,
        target_temperature=target_temperature,
        std=10.0
    )
    
    action_penalty = compute_action_penalty(
        action=action,
    )
    
    reward = TEMPERATURE_ERROR_COEF * temperature_error + \
             TEMPERATURE_ERROR_TANH_COEF * temp_error_tanh + \
             ACTION_PENALTY_COEF * action_penalty
    print(f"reward: {reward}")
    return reward