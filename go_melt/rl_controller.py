"""
RL Controller Module for GO-MELT Power Control

This module provides functions for reinforcement learning-based power control.
Currently implements a simple linear controller as a placeholder for future RL agent.

The controller adjusts laser power at regular intervals (every 10 steps = 0.1m)
based on the maximum temperature observed in Level 3, with the goal of maintaining
the maximum temperature around a target value (max 3000K).
"""

import jax.numpy as jnp
import numpy as np
from typing import Dict, Tuple, Optional, Union, List


def get_max_temperature_level3(Levels) -> float:
    """
    Extract the maximum temperature from Level 3 temperature field.
    
    This serves as the observation for the RL controller.
    
    Parameters:
    -----------
    Levels : list or dict
        List or dictionary containing all level data. Must have Levels[3]["T0"]
        which is the temperature field for Level 3.
    
    Returns:
    --------
    float
        Maximum temperature in Level 3 (Kelvin)
    """
    # Check if Levels is a list or dict and if Level 3 exists
    if isinstance(Levels, list):
        if len(Levels) < 4:
            raise ValueError("Level 3 not found: Levels list has fewer than 4 elements")
        level3 = Levels[3]
    elif isinstance(Levels, dict):
        if 3 not in Levels:
            raise ValueError("Level 3 not found in Levels dictionary")
        level3 = Levels[3]
    else:
        raise ValueError("Levels must be a list or dict")
    
    if not isinstance(level3, dict) or "T0" not in level3:
        raise ValueError("Temperature field T0 not found in Level 3")
    
    # Get temperature array (convert JAX array to numpy if needed)
    temp_field = level3["T0"]
    
    # Handle both JAX and numpy arrays
    if hasattr(temp_field, 'block_until_ready'):
        # JAX array - compute max
        max_temp = float(jnp.max(temp_field))
    else:
        # NumPy array
        max_temp = float(np.max(temp_field))
    
    return max_temp


def linear_power_controller(
    max_temperature: float,
    target_temperature: float = 3000.0,
    current_power: float = 285.0,
    kp: float = 0.1,
    power_min: float = 0.0,
    power_max: float = 500.0
) -> float:
    """
    Simple linear proportional controller for power adjustment.
    
    This is a placeholder that will be replaced by an RL agent.
    Uses a proportional control law: power_adjustment = kp * (target - current_temp)
    
    Parameters:
    -----------
    max_temperature : float
        Current maximum temperature in Level 3 (Kelvin) - the observation
    target_temperature : float
        Target maximum temperature (Kelvin). Default: 3000.0
    current_power : float
        Current power level (Watts). Default: 285.0
    kp : float
        Proportional gain. Positive value means higher temp -> lower power.
        Default: 0.1 (W/K)
    power_min : float
        Minimum allowed power (Watts). Default: 0.0
    power_max : float
        Maximum allowed power (Watts). Default: 500.0
    
    Returns:
    --------
    float
        Adjusted power value (Watts), clamped to [power_min, power_max]
    """
    # Calculate temperature error
    temp_error = target_temperature - max_temperature
    
    # Proportional control: adjust power based on temperature error
    # If temp is too high (error < 0), reduce power (negative adjustment)
    # If temp is too low (error > 0), increase power (positive adjustment)
    power_adjustment = kp * temp_error
    
    # Calculate new power by adjusting from current power
    new_power = current_power + power_adjustment
    
    # Clamp to valid range
    new_power = np.clip(new_power, power_min, power_max)
    
    return float(new_power)


def should_update_power(
    step_count: int,
    update_interval: int = 10
) -> bool:
    """
    Check if power should be updated at this step.
    
    Parameters:
    -----------
    step_count : int
        Current simulation step count
    update_interval : int
        Number of steps between power updates. Default: 10 (every 10 steps = 0.1m)
    
    Returns:
    --------
    bool
        True if power should be updated, False otherwise
    """
    return (step_count % update_interval) == 0


def compute_power_action(
    observation: float,
    target_temperature: float = 3000.0,
    current_power: Optional[float] = None,
    base_power: float = 285.0,
    controller_params: Optional[Dict] = None
) -> float:
    """
    Compute power action based on observation (max temperature).
    
    This is the main interface that will be replaced by an RL agent.
    Currently uses linear controller, but structure allows easy replacement.
    
    Parameters:
    -----------
    observation : float
        Maximum temperature from Level 3 (Kelvin)
    target_temperature : float
        Target maximum temperature (Kelvin). Default: 3000.0
    current_power : float, optional
        Current power level. If None, uses base_power. Default: None
    base_power : float
        Base power level (Watts). Used if current_power is None. Default: 285.0
    controller_params : dict, optional
        Additional parameters for the controller. Default: None
        Can include: kp, power_min, power_max
    
    Returns:
    --------
    float
        New power value (Watts)
    """
    # Use current_power or fall back to base_power
    if current_power is None:
        current_power = base_power
    
    # Default controller parameters
    if controller_params is None:
        controller_params = {}
    
    kp = controller_params.get('kp', 0.1)
    power_min = controller_params.get('power_min', 0.0)
    power_max = controller_params.get('power_max', 500.0)
    
    # Call linear controller
    new_power = linear_power_controller(
        max_temperature=observation,
        target_temperature=target_temperature,
        current_power=current_power,
        kp=kp,
        power_min=power_min,
        power_max=power_max
    )
    
    return new_power


def get_observation(Levels: Union[List, Dict]) -> float:
    """
    Get observation for RL controller (max temperature in Level 3).
    
    This is a convenience wrapper around get_max_temperature_level3()
    to match RL terminology.
    
    Parameters:
    -----------
    Levels : list or dict
        List or dictionary containing all level data
    
    Returns:
    --------
    float
        Observation (maximum temperature in Level 3, Kelvin)
    """
    return get_max_temperature_level3(Levels)


class PowerController:
    """
    Power Controller class for managing power updates during simulation.
    
    This class tracks step count and manages power adjustments at regular intervals.
    """
    
    def __init__(
        self,
        target_temperature: float = 3000.0,
        base_power: float = 285.0,
        update_interval: int = 10,
        controller_params: Optional[Dict] = None
    ):
        """
        Initialize the power controller.
        
        Parameters:
        -----------
        target_temperature : float
            Target maximum temperature (Kelvin). Default: 3000.0
        base_power : float
            Initial/base power level (Watts). Default: 285.0
        update_interval : int
            Number of steps between power updates. Default: 10 (0.1m)
        controller_params : dict, optional
            Controller parameters (kp, power_min, power_max). Default: None
        """
        self.target_temperature = target_temperature
        self.current_power = base_power
        self.base_power = base_power
        self.update_interval = update_interval
        self.step_count = 0
        self.controller_params = controller_params or {}
        
        # Store history for potential use in RL training
        self.observation_history = []
        self.action_history = []
        self.power_history = []
    
    def step(self, Levels: Union[List, Dict]) -> Tuple[float, bool]:
        """
        Execute one control step.
        
        Parameters:
        -----------
        Levels : list or dict
            List or dictionary containing all level data
        
        Returns:
        --------
        tuple (float, bool)
            - New power value (Watts)
            - Whether power was updated this step (bool)
        """
        self.step_count += 1
        should_update = should_update_power(self.step_count, self.update_interval)
        
        if should_update:
            # Get observation
            observation = get_observation(Levels)
            
            # Compute action (new power)
            new_power = compute_power_action(
                observation=observation,
                target_temperature=self.target_temperature,
                current_power=self.current_power,
                base_power=self.base_power,
                controller_params=self.controller_params
            )
            
            # Update current power
            self.current_power = new_power
            
            # Store history
            self.observation_history.append(observation)
            self.action_history.append(new_power)
            self.power_history.append(self.current_power)
            
            return new_power, True
        else:
            # No update, return current power
            return self.current_power, False
    
    def get_current_power(self) -> float:
        """Get the current power value."""
        return self.current_power
    
    def reset(self, base_power: Optional[float] = None):
        """Reset the controller (clear history, reset step count)."""
        if base_power is not None:
            self.current_power = base_power
            self.base_power = base_power
        else:
            self.current_power = self.base_power
        
        self.step_count = 0
        self.observation_history.clear()
        self.action_history.clear()
        self.power_history.clear()
    
    def get_history(self) -> Dict:
        """Get control history for analysis/RL training."""
        return {
            'observations': self.observation_history.copy(),
            'actions': self.action_history.copy(),
            'power': self.power_history.copy(),
            'steps': list(range(0, len(self.observation_history) * self.update_interval, self.update_interval))
        }

