"""
Power Controller Module for GO-MELT Power Control

This module provides controllers for power control:
- PID Controller: Proportional-Integral-Derivative controller
- RL Controller: Reinforcement learning-based controller (currently uses linear controller as placeholder)

The controllers adjust laser power at regular intervals (every 10 steps = 0.1m)
based on the maximum temperature observed in Level 3, with the goal of maintaining
the maximum temperature around a target value (max 3000K).
"""

import jax.numpy as jnp
import numpy as np
from typing import Dict, Tuple, Optional, Union, List
import os


def get_max_temperature_level3(Levels: Union[List, Dict]) -> float:
    """
    Extract the maximum temperature from Level 3 temperature field.
    
    This serves as the observation for the controllers.
    
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
    Get observation for controllers (max temperature in Level 3).
    
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


class PIDController:
    """
    PID (Proportional-Integral-Derivative) Controller for power control.
    
    This class implements a PID controller to maintain target temperature
    by adjusting laser power based on temperature error.
    """
    
    def __init__(
        self,
        target_temperature: float = 3000.0,
        base_power: float = 285.0,
        update_interval: int = 10,
        controller_params: Optional[Dict] = None
    ):
        """
        Initialize the PID controller.
        
        Parameters:
        -----------
        target_temperature : float
            Target maximum temperature (Kelvin). Default: 3000.0
        base_power : float
            Initial/base power level (Watts). Default: 285.0
        update_interval : int
            Number of steps between power updates. Default: 10 (0.1m)
        controller_params : dict, optional
            Controller parameters. Default: None
            Required keys:
                - kp: Proportional gain (W/K). Default: 0.1
                - ki: Integral gain (W/(K·s)). Default: 0.01
                - kd: Derivative gain (W·s/K). Default: 0.0
                - power_min: Minimum allowed power (W). Default: 0.0
                - power_max: Maximum allowed power (W). Default: 500.0
                - integral_limit: Maximum integral term (W). Default: 100.0
        """
        self.target_temperature = target_temperature
        self.current_power = base_power
        self.base_power = base_power
        self.update_interval = update_interval
        self.step_count = 0
        
        # Parse controller parameters
        params = controller_params or {}
        self.kp = params.get('kp', 0.1)
        self.ki = params.get('ki', 0.01)
        self.kd = params.get('kd', 0.0)
        self.power_min = params.get('power_min', 0.0)
        self.power_max = params.get('power_max', 500.0)
        self.integral_limit = params.get('integral_limit', 100.0)
        
        # PID state variables
        self.integral_error = 0.0
        self.prev_error = 0.0
        self.prev_time = None
        
        # Store history
        self.observation_history = []
        self.action_history = []
        self.power_history = []
        self.error_history = []
    
    def _compute_pid_output(self, error: float, dt: float = 1.0) -> float:
        """
        Compute PID controller output.
        
        Parameters:
        -----------
        error : float
            Temperature error (target - current)
        dt : float
            Time step (for integral and derivative terms). Default: 1.0
        
        Returns:
        --------
        float
            Power adjustment (Watts)
        """
        # Proportional term
        p_term = self.kp * error
        
        # Integral term (with anti-windup)
        self.integral_error += error * dt
        # Limit integral term to prevent windup
        self.integral_error = np.clip(self.integral_error, -self.integral_limit, self.integral_limit)
        i_term = self.ki * self.integral_error
        
        # Derivative term
        if self.prev_time is not None and dt > 0:
            d_error = (error - self.prev_error) / dt
            d_term = self.kd * d_error
        else:
            d_term = 0.0
        
        # Update previous values
        self.prev_error = error
        self.prev_time = dt if self.prev_time is None else self.prev_time + dt
        
        # Total PID output
        pid_output = p_term + i_term + d_term
        
        return pid_output
    
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
            # Get observation (max temperature)
            observation = get_max_temperature_level3(Levels)
            
            # Calculate error (target - current)
            error = self.target_temperature - observation
            
            # Compute PID output (power adjustment)
            # Use update_interval as time step for integral/derivative
            dt = float(self.update_interval)  # Approximate time step
            power_adjustment = self._compute_pid_output(error, dt)
            
            # Calculate new power
            new_power = self.current_power + power_adjustment
            
            # Clamp to valid range
            new_power = np.clip(new_power, self.power_min, self.power_max)
            
            # Update current power
            self.current_power = new_power
            
            # Store history
            self.observation_history.append(observation)
            self.action_history.append(new_power)
            self.power_history.append(self.current_power)
            self.error_history.append(error)
            
            return new_power, True
        else:
            # No update, return current power
            return self.current_power, False
    
    def get_current_power(self) -> float:
        """Get the current power value."""
        return self.current_power
    
    def reset(self, base_power: Optional[float] = None):
        """Reset the controller (clear history, reset step count and PID state)."""
        if base_power is not None:
            self.current_power = base_power
            self.base_power = base_power
        else:
            self.current_power = self.base_power
        
        self.step_count = 0
        self.integral_error = 0.0
        self.prev_error = 0.0
        self.prev_time = None
        self.observation_history.clear()
        self.action_history.clear()
        self.power_history.clear()
        self.error_history.clear()
    
    def get_history(self) -> Dict:
        """Get control history for analysis."""
        return {
            'observations': self.observation_history.copy(),
            'actions': self.action_history.copy(),
            'power': self.power_history.copy(),
            'errors': self.error_history.copy(),
            'steps': list(range(0, len(self.observation_history) * self.update_interval, self.update_interval))
        }


class PowerController:
    """
    RL Power Controller class for managing power updates during simulation.
    
    This class tracks step count and manages power adjustments at regular intervals.
    Can use either a linear controller (default) or a trained RL agent.
    """
    
    def __init__(
        self,
        target_temperature: float = 3000.0,
        base_power: float = 285.0,
        update_interval: int = 10,
        controller_params: Optional[Dict] = None,
        rl_model_path: Optional[str] = None
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
        rl_model_path : str, optional
            Path to trained RL model file (.zip). If provided, uses RL agent instead of linear controller.
            Default: None (uses linear controller)
        """
        self.target_temperature = target_temperature
        self.current_power = base_power
        self.base_power = base_power
        self.update_interval = update_interval
        self.step_count = 0
        self.controller_params = controller_params or {}
        
        # RL model (if provided)
        self.rl_model = None
        self.use_rl = False
        self.temp_min = 0.0
        self.temp_max = 5000.0
        
        if rl_model_path and os.path.exists(rl_model_path):
            try:
                from stable_baselines3 import PPO, SAC, TD3
                
                # Try to load the model (will auto-detect algorithm)
                # First try PPO, then SAC, then TD3
                try:
                    self.rl_model = PPO.load(rl_model_path)
                    self.use_rl = True
                    print(f"Loaded RL model (PPO) from: {rl_model_path}")
                except:
                    try:
                        self.rl_model = SAC.load(rl_model_path)
                        self.use_rl = True
                        print(f"Loaded RL model (SAC) from: {rl_model_path}")
                    except:
                        try:
                            self.rl_model = TD3.load(rl_model_path)
                            self.use_rl = True
                            print(f"Loaded RL model (TD3) from: {rl_model_path}")
                        except Exception as e:
                            print(f"Warning: Failed to load RL model from {rl_model_path}: {e}")
                            print("Falling back to linear controller.")
                            self.rl_model = None
                            self.use_rl = False
            except ImportError:
                print("Warning: stable-baselines3 not installed. Cannot load RL model.")
                print("Install with: pip install stable-baselines3")
                self.rl_model = None
                self.use_rl = False
        elif rl_model_path:
            print(f"Warning: RL model path not found: {rl_model_path}")
            print("Falling back to linear controller.")
        
        # Store history for potential use in RL training
        self.observation_history = []
        self.action_history = []
        self.power_history = []
    
    def _normalize_temperature(self, temp: float) -> float:
        """Normalize temperature to [0, 1] range for RL agent."""
        return np.clip((temp - self.temp_min) / (self.temp_max - self.temp_min), 0.0, 1.0)
    
    def _action_to_power(self, action: np.ndarray) -> float:
        """
        Convert normalized RL action [-1, 1] to power value.
        
        Parameters:
        -----------
        action : np.ndarray
            Normalized action from RL agent
            
        Returns:
        --------
        float
            Power value in Watts
        """
        power_min = self.controller_params.get('power_min', 0.0)
        power_max = self.controller_params.get('power_max', 500.0)
        power_range = power_max - power_min
        
        # Action is power adjustment (relative to current)
        power_adjustment = float(action[0]) * power_range * 0.01  # Scale to ±50% of range
        new_power = self.current_power + power_adjustment
        
        # Clamp to valid range
        new_power = np.clip(new_power, power_min, power_max)
        return float(new_power)
    
    def step(self, Levels: Union[List, Dict]) -> Tuple[float, bool]:
        """
        Execute one control step.
        
        Parameters:
        -----------
        Levels : dict
            Dictionary containing all level data
        
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
            if self.use_rl and self.rl_model is not None:
                # Use RL agent
                normalized_obs = np.array([self._normalize_temperature(observation)], dtype=np.float32)
                action, _ = self.rl_model.predict(normalized_obs, deterministic=True)
                new_power = self._action_to_power(action)
            else:
                # Use linear controller
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

