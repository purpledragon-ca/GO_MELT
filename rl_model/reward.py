"""
Reward Computation for GO-MELT RL Environment

This module provides reward computation functions for reinforcement learning
in the GO-MELT simulation environment. It includes:
- Temperature-based reward: based on error from target temperature
- Power change reward: based on power change amount to encourage smooth control
- Combined reward: balances temperature accuracy and power change smoothness with configurable weights
"""

import numpy as np


def compute_reward(
    temperature: float,
    target_temperature: float = 3000.0,
    reward_scale: float = 0.1,
    reward_min: float = -100.0,
    reward_max: float = 0.0
) -> float:
    """
    Compute reward based on temperature error.
    
    Reward is negative squared error from target temperature, normalized.
    This encourages the agent to maintain temperature close to the target.
    
    Parameters:
    -----------
    temperature : float
        Current maximum temperature (Kelvin)
    target_temperature : float
        Target maximum temperature (Kelvin). Default: 3000.0
    reward_scale : float
        Scaling factor for reward. Default: 0.1
        Higher values make the reward more sensitive to errors.
    reward_min : float
        Minimum reward value (clipping). Default: -100.0
    reward_max : float
        Maximum reward value (clipping). Default: 0.0
    
    Returns:
    --------
    float
        Reward value (typically negative, with 0 being the best)
    
    Examples:
    ---------
    >>> # Perfect temperature match
    >>> reward = compute_reward(3000.0, 3000.0)
    >>> print(reward)  # 0.0
    
    >>> # Small error
    >>> reward = compute_reward(2950.0, 3000.0)
    >>> print(reward)  # Small negative value
    
    >>> # Large error
    >>> reward = compute_reward(2500.0, 3000.0)
    >>> print(reward)  # Larger negative value
    """
    error = target_temperature - temperature
    
    # Normalize error by target temperature to keep rewards in reasonable range
    normalized_error = error / target_temperature
    
    # Scale reward to be in range [-1, 0] when error is small
    # Use smaller scale to prevent explosion
    reward = -reward_scale * (normalized_error ** 2)
    
    # Clip reward to prevent extreme values
    reward = np.clip(reward, reward_min, reward_max)
    
    return float(reward)


def compute_reward_power_change(
    power_change: float,
    power_min: float = 0.0,
    power_max: float = 500.0,
    reward_scale: float = 0.01,
    reward_min: float = -100.0,
    reward_max: float = 0.0
) -> float:
    """
    Compute reward based on power change amount.
    
    This reward penalizes large power changes to encourage smoother control.
    Smaller power adjustments result in higher (less negative) rewards.
    
    Parameters:
    -----------
    power_change : float
        Change in power from previous step (Watts). Can be positive or negative.
    power_min : float
        Minimum allowed power (Watts). Default: 0.0
        Used for normalization.
    power_max : float
        Maximum allowed power (Watts). Default: 500.0
        Used for normalization.
    reward_scale : float
        Scaling factor for reward. Default: 0.01
        Higher values make the reward more sensitive to power changes.
    reward_min : float
        Minimum reward value (clipping). Default: -100.0
    reward_max : float
        Maximum reward value (clipping). Default: 0.0
    
    Returns:
    --------
    float
        Reward value (typically negative, with 0 being the best for no change)
    
    Examples:
    ---------
    >>> # No power change (smooth control)
    >>> reward = compute_reward_power_change(0.0)
    >>> print(reward)  # 0.0
    
    >>> # Small power change
    >>> reward = compute_reward_power_change(10.0)
    >>> print(reward)  # Small negative value
    
    >>> # Large power change
    >>> reward = compute_reward_power_change(100.0)
    >>> print(reward)  # Larger negative value
    """
    power_range = power_max - power_min
    
    # Normalize power change by power range to keep rewards in reasonable range
    normalized_change = abs(power_change) / power_range
    
    # Penalize power changes (squared to penalize large changes more)
    # Reward is negative, with 0 being the best (no change)
    reward = -reward_scale * (normalized_change ** 2)
    
    # Clip reward to prevent extreme values
    reward = np.clip(reward, reward_min, reward_max)
    
    return float(reward)


def compute_combined_reward(
    temperature: float,
    power_change: float,
    target_temperature: float = 3000.0,
    power_min: float = 0.0,
    power_max: float = 500.0,
    temperature_weight: float = 0.8,
    power_change_weight: float = 0.2,
    temperature_reward_scale: float = 0.1,
    power_change_reward_scale: float = 0.01,
    reward_min: float = -100.0,
    reward_max: float = 0.0
) -> float:
    """
    Compute combined reward that balances temperature accuracy and power change smoothness.
    
    This function combines two reward components:
    1. Temperature reward: encourages maintaining target temperature
    2. Power change reward: encourages smooth control (limits power changes)
    
    The weights allow you to prioritize one objective over the other.
    Default weights (0.8 temperature, 0.2 power change) prioritize temperature accuracy
    while still encouraging smooth control.
    
    Parameters:
    -----------
    temperature : float
        Current maximum temperature (Kelvin)
    power_change : float
        Change in power from previous step (Watts). Can be positive or negative.
    target_temperature : float
        Target maximum temperature (Kelvin). Default: 3000.0
    power_min : float
        Minimum allowed power (Watts). Default: 0.0
    power_max : float
        Maximum allowed power (Watts). Default: 500.0
    temperature_weight : float
        Weight for temperature reward component. Default: 0.8
        Higher values prioritize temperature accuracy.
        Should be in range [0, 1]. temperature_weight + power_change_weight should sum to 1.0.
    power_change_weight : float
        Weight for power change reward component. Default: 0.2
        Higher values prioritize smooth control (smaller power changes).
        Should be in range [0, 1]. temperature_weight + power_change_weight should sum to 1.0.
    temperature_reward_scale : float
        Scaling factor for temperature reward. Default: 0.1
        Higher values make temperature reward more sensitive to errors.
    power_change_reward_scale : float
        Scaling factor for power change reward. Default: 0.01
        Higher values make power change reward more sensitive to changes.
    reward_min : float
        Minimum reward value (clipping). Default: -100.0
    reward_max : float
        Maximum reward value (clipping). Default: 0.0
    
    Returns:
    --------
    float
        Combined reward value (typically negative, with 0 being the best)
    
    Examples:
    ---------
    >>> # Perfect temperature, no power change
    >>> reward = compute_combined_reward(3000.0, 0.0, 3000.0)
    >>> print(reward)  # 0.0
    
    >>> # Good temperature, small power change
    >>> reward = compute_combined_reward(2990.0, 5.0, 3000.0)
    >>> print(reward)  # Small negative value
    
    >>> # Poor temperature, large power change
    >>> reward = compute_combined_reward(2800.0, 100.0, 3000.0)
    >>> print(reward)  # Large negative value
    
    >>> # Prioritize temperature accuracy (80% weight)
    >>> reward = compute_combined_reward(
    ...     temperature=2950.0,
    ...     power_change=50.0,
    ...     target_temperature=3000.0,
    ...     temperature_weight=0.8,
    ...     power_change_weight=0.2
    ... )
    
    >>> # Prioritize smooth control (40% temperature, 60% power change)
    >>> reward = compute_combined_reward(
    ...     temperature=2950.0,
    ...     power_change=50.0,
    ...     target_temperature=3000.0,
    ...     temperature_weight=0.4,
    ...     power_change_weight=0.6
    ... )
    """
    # Compute temperature reward component
    temp_reward = compute_reward(
        temperature=temperature,
        target_temperature=target_temperature,
        reward_scale=temperature_reward_scale,
        reward_min=reward_min,
        reward_max=reward_max
    )
    
    # Compute power change reward component
    power_reward = compute_reward_power_change(
        power_change=power_change,
        power_min=power_min,
        power_max=power_max,
        reward_scale=power_change_reward_scale,
        reward_min=reward_min,
        reward_max=reward_max
    )
    
    # Combine rewards with weights
    # Normalize weights to ensure they sum to 1.0
    total_weight = temperature_weight + power_change_weight
    if total_weight > 0:
        normalized_temp_weight = temperature_weight / total_weight
        normalized_power_weight = power_change_weight / total_weight
    else:
        # If both weights are 0, use equal weights
        normalized_temp_weight = 0.5
        normalized_power_weight = 0.5
    
    combined_reward = (normalized_temp_weight * temp_reward + 
                      normalized_power_weight * power_reward)
    
    # Clip reward to prevent extreme values
    combined_reward = np.clip(combined_reward, reward_min, reward_max)
    
    return float(combined_reward)
