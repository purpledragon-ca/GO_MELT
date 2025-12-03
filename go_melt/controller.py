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


def _get_level3_temp_field(Levels: Union[List, Dict]):
    """Helper to extract Level 3 temperature field."""
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
    
    return level3["T0"]


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
    temp_field = _get_level3_temp_field(Levels)
    
    # Handle both JAX and numpy arrays
    if hasattr(temp_field, 'block_until_ready'):
        # JAX array - compute max
        max_temp = float(jnp.max(temp_field))
    else:
        # NumPy array
        max_temp = float(np.max(temp_field))
    
    return max_temp


def get_temperature_stats_level3(Levels: Union[List, Dict]) -> Dict[str, float]:
    """
    Extract temperature statistics from Level 3.
    
    Parameters:
    -----------
    Levels : list or dict
        List or dictionary containing all level data
    
    Returns:
    --------
    dict
        Dictionary with keys: 'max', 'min', 'avg', 'std'
    """
    temp_field = _get_level3_temp_field(Levels)
    
    # Handle both JAX and numpy arrays
    if hasattr(temp_field, 'block_until_ready'):
        # JAX array
        max_temp = float(jnp.max(temp_field))
        min_temp = float(jnp.min(temp_field))
        avg_temp = float(jnp.mean(temp_field))
        std_temp = float(jnp.std(temp_field))
    else:
        # NumPy array
        max_temp = float(np.max(temp_field))
        min_temp = float(np.min(temp_field))
        avg_temp = float(np.mean(temp_field))
        std_temp = float(np.std(temp_field))
    
    return {
        'max': max_temp,
        'min': min_temp,
        'avg': avg_temp,
        'std': std_temp
    }


def get_fraction_above_liquidus(Levels: Union[List, Dict], T_liquidus: float) -> float:
    """
    Compute fraction of Level 3 cells above liquidus temperature (melt pool size proxy).
    
    Parameters:
    -----------
    Levels : list or dict
        List or dictionary containing all level data
    T_liquidus : float
        Liquidus temperature (Kelvin)
    
    Returns:
    --------
    float
        Fraction of cells above liquidus (0.0 to 1.0)
    """
    temp_field = _get_level3_temp_field(Levels)
    
    # Handle both JAX and numpy arrays
    if hasattr(temp_field, 'block_until_ready'):
        # JAX array
        above_liquidus = jnp.sum(temp_field > T_liquidus)
        total_cells = temp_field.size
        fraction = float(above_liquidus / total_cells)
    else:
        # NumPy array
        above_liquidus = np.sum(temp_field > T_liquidus)
        total_cells = temp_field.size
        fraction = float(above_liquidus / total_cells)
    
    return fraction


def get_accum_time_local(accum_time: Optional[np.ndarray], idx: Optional[int] = None) -> float:
    """
    Get local accumulated time at specified index.
    
    Parameters:
    -----------
    accum_time : np.ndarray, optional
        Accumulated time array
    idx : int, optional
        Index to get time at. If None, returns max value.
    
    Returns:
    --------
    float
        Accumulated time (seconds)
    """
    if accum_time is None:
        return 0.0
    
    if hasattr(accum_time, 'block_until_ready'):
        # JAX array
        if idx is not None:
            return float(accum_time[idx])
        else:
            return float(jnp.max(accum_time))
    else:
        # NumPy array
        if idx is not None:
            return float(accum_time[idx])
        else:
            return float(np.max(accum_time))


def get_laser_speed(laser_all: Optional[np.ndarray]) -> float:
    """
    Get current laser speed from toolpath.
    
    Parameters:
    -----------
    laser_all : np.ndarray, optional
        Array of laser positions [x, y, z, jump, dwell, dt, power]
        dt is the time step, speed can be computed from position changes
    
    Returns:
    --------
    float
        Laser speed (mm/s or units/s)
    """
    if laser_all is None or len(laser_all) < 2:
        return 0.0
    
    # Get last two positions
    if hasattr(laser_all, 'block_until_ready'):
        # JAX array
        pos1 = laser_all[-2, :3]
        pos2 = laser_all[-1, :3]
        dt = float(laser_all[-1, 5])
        distance = float(jnp.linalg.norm(pos2 - pos1))
    else:
        # NumPy array
        pos1 = laser_all[-2, :3]
        pos2 = laser_all[-1, :3]
        dt = float(laser_all[-1, 5])
        distance = float(np.linalg.norm(pos2 - pos1))
    
    if dt > 0:
        speed = distance / dt
    else:
        speed = 0.0
    
    return speed


def get_distance_to_last_point(laser_all: Optional[np.ndarray]) -> float:
    """
    Get distance from current position to last point in toolpath.
    
    Parameters:
    -----------
    laser_all : np.ndarray, optional
        Array of laser positions
    
    Returns:
    --------
    float
        Distance to last point
    """
    if laser_all is None or len(laser_all) < 2:
        return 0.0
    
    if hasattr(laser_all, 'block_until_ready'):
        # JAX array
        current_pos = laser_all[-1, :3]
        last_pos = laser_all[-2, :3]
        distance = float(jnp.linalg.norm(current_pos - last_pos))
    else:
        # NumPy array
        current_pos = laser_all[-1, :3]
        last_pos = laser_all[-2, :3]
        distance = float(np.linalg.norm(current_pos - last_pos))
    
    return distance


def get_relative_toolpath(laser_all: Optional[np.ndarray], length: int, future: bool = True) -> np.ndarray:
    """
    Get relative toolpath (future or history) translated from current position.
    
    Parameters:
    -----------
    laser_all : np.ndarray, optional
        Array of laser positions [x, y, z, jump, dwell, dt, power]
    length : int
        Number of points to include
    future : bool
        If True, get future points; if False, get history points
    
    Returns:
    --------
    np.ndarray
        Relative path array of shape (length, 3) with [x, y, z] relative to current position
        Padded with zeros if insufficient points
    """
    if laser_all is None or len(laser_all) == 0:
        return np.zeros((length, 3), dtype=np.float32)
    
    # Get current position
    if hasattr(laser_all, 'block_until_ready'):
        # JAX array
        current_pos = laser_all[-1, :3]
        if future:
            # For future, we'd need to read ahead - for now return zeros
            # In practice, this would need access to future toolpath
            return np.zeros((length, 3), dtype=np.float32)
        else:
            # Get history points
            start_idx = max(0, len(laser_all) - length - 1)
            history_positions = laser_all[start_idx:-1, :3]  # Exclude current
            relative_path = history_positions - current_pos
            # Convert to numpy and pad if needed
            relative_path = np.array(relative_path, dtype=np.float32)
    else:
        # NumPy array
        current_pos = laser_all[-1, :3]
        if future:
            # For future, return zeros (would need toolpath file access)
            return np.zeros((length, 3), dtype=np.float32)
        else:
            # Get history points
            start_idx = max(0, len(laser_all) - length - 1)
            history_positions = laser_all[start_idx:-1, :3]  # Exclude current
            relative_path = history_positions - current_pos
    
    # Pad if needed
    if len(relative_path) < length:
        padding = np.zeros((length - len(relative_path), 3), dtype=np.float32)
        relative_path = np.concatenate([padding, relative_path], axis=0)
    
    # Take only the requested length
    return relative_path[-length:] if len(relative_path) > length else relative_path


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


def get_observation(
    Levels: Union[List, Dict],
    observation_config: Optional[Dict] = None,
    power_history: Optional[List[float]] = None,
    temperature_history: Optional[List[float]] = None,
    laser_all: Optional[np.ndarray] = None,
    accum_time: Optional[np.ndarray] = None,
    accum_idx: Optional[int] = None,
    Properties: Optional[Dict] = None,
    toolpath_history: Optional[List[np.ndarray]] = None,
    toolpath_future: Optional[List[np.ndarray]] = None,
    power_history_toolpath: Optional[List[float]] = None,
    # Legacy parameters for backward compatibility
    history_length: int = 10,
    power_min: float = 0.0,
    power_max: float = 500.0,
    temp_min: float = 0.0,
    temp_max: float = 5000.0
) -> np.ndarray:
    """
    Get observation for controllers with configurable features.
    
    Builds observation array dynamically based on observation_config.
    All features can be enabled/disabled via config.
    
    Parameters:
    -----------
    Levels : list or dict
        List or dictionary containing all level data
    observation_config : dict, optional
        Configuration dict specifying which features to include. If None, uses defaults.
        Keys:
            - enable_max_T_L3: bool (default True)
            - enable_avg_T_L3: bool (default False)
            - enable_min_T_L3: bool (default False)
            - enable_std_T_L3: bool (default False)
            - enable_frac_above_liquidus: bool (default False)
            - enable_accum_time_local: bool (default False)
            - enable_laser_speed: bool (default False)
            - enable_distance_to_last_point: bool (default False)
            - enable_current_power: bool (default True)
            - enable_power_history: bool (default True)
            - enable_temperature_history: bool (default True)
            - enable_future_toolpath: bool (default False)
            - enable_history_toolpath: bool (default False)
            - power_history_length: int (default 10)
            - temperature_history_length: int (default 10)
            - future_toolpath_length: int (default 10)
            - history_toolpath_length: int (default 10)
            - power_min, power_max, temp_min, temp_max: normalization ranges
    power_history : list of float, optional
        History of power values
    temperature_history : list of float, optional
        History of temperature values
    laser_all : np.ndarray, optional
        Array of laser positions [x, y, z, jump, dwell, dt, power]
    accum_time : np.ndarray, optional
        Accumulated time array
    accum_idx : int, optional
        Index for local accumulated time
    Properties : dict, optional
        Properties dict containing T_liquidus
    toolpath_history : list of np.ndarray, optional
        History of toolpath positions (relative)
    toolpath_future : list of np.ndarray, optional
        Future toolpath positions (relative)
    power_history_toolpath : list of float, optional
        Power history corresponding to toolpath_history
    
    Returns:
    --------
    np.ndarray
        Observation array with selected features (dynamically sized)
    """
    # Default observation config
    if observation_config is None:
        observation_config = {}
    
    # Get normalization ranges
    obs_power_min = observation_config.get('power_min', power_min)
    obs_power_max = observation_config.get('power_max', power_max)
    obs_temp_min = observation_config.get('temp_min', temp_min)
    obs_temp_max = observation_config.get('temp_max', temp_max)
    
    # Get history lengths
    power_hist_len = observation_config.get('power_history_length', history_length)
    temp_hist_len = observation_config.get('temperature_history_length', history_length)
    future_toolpath_len = observation_config.get('future_toolpath_length', 10)
    history_toolpath_len = observation_config.get('history_toolpath_length', 10)
    
    # Build observation array
    obs_list = []
    
    # Temperature statistics
    if observation_config.get('enable_max_T_L3', True):
        max_temp = get_max_temperature_level3(Levels)
        normalized = float(np.clip((max_temp - obs_temp_min) / (obs_temp_max - obs_temp_min), 0.0, 1.0))
        obs_list.append(normalized)
    
    if observation_config.get('enable_avg_T_L3', False):
        temp_stats = get_temperature_stats_level3(Levels)
        normalized = float(np.clip((temp_stats['avg'] - obs_temp_min) / (obs_temp_max - obs_temp_min), 0.0, 1.0))
        obs_list.append(normalized)
    
    if observation_config.get('enable_min_T_L3', False):
        temp_stats = get_temperature_stats_level3(Levels)
        normalized = float(np.clip((temp_stats['min'] - obs_temp_min) / (obs_temp_max - obs_temp_min), 0.0, 1.0))
        obs_list.append(normalized)
    
    if observation_config.get('enable_std_T_L3', False):
        temp_stats = get_temperature_stats_level3(Levels)
        # Normalize std by temp range
        normalized = float(np.clip(temp_stats['std'] / (obs_temp_max - obs_temp_min), 0.0, 1.0))
        obs_list.append(normalized)
    
    # Fraction above liquidus
    if observation_config.get('enable_frac_above_liquidus', False):
        if Properties is not None and 'T_liquidus' in Properties:
            frac = float(get_fraction_above_liquidus(Levels, Properties['T_liquidus']))
            obs_list.append(frac)  # Already 0-1
        else:
            obs_list.append(0.0)
    
    # Accumulated time local
    if observation_config.get('enable_accum_time_local', False):
        accum_local = get_accum_time_local(accum_time, accum_idx)
        # Normalize by reasonable max (e.g., 1 second)
        normalized = float(np.clip(accum_local / 1.0, 0.0, 1.0))
        obs_list.append(normalized)
    
    # Laser speed
    if observation_config.get('enable_laser_speed', False):
        speed = get_laser_speed(laser_all)
        # Normalize by typical max speed (e.g., 2000 mm/s)
        normalized = float(np.clip(speed / 2000.0, 0.0, 1.0))
        obs_list.append(normalized)
    
    # Distance to last point
    if observation_config.get('enable_distance_to_last_point', False):
        distance = get_distance_to_last_point(laser_all)
        # Normalize by typical max distance (e.g., 1 mm)
        normalized = float(np.clip(distance / 1.0, 0.0, 1.0))
        obs_list.append(normalized)
    
    # Current power (removed - not used in observation)
    # if observation_config.get('enable_current_power', True):
    #     if power_history and len(power_history) > 0:
    #         current_power = power_history[-1]
    #     else:
    #         current_power = 0.0
    #     normalized = np.clip((current_power - obs_power_min) / (obs_power_max - obs_power_min), 0.0, 1.0)
    #     obs_list.append(normalized)
    
    # Power history
    if observation_config.get('enable_power_history', True):
        power_hist = power_history or []
        normalized_power_hist = []
        for p in power_hist[-power_hist_len:]:
            # Ensure p is a scalar
            p_val = float(p) if hasattr(p, '__float__') else float(p)
            normalized_p = float(np.clip((p_val - obs_power_min) / (obs_power_max - obs_power_min), 0.0, 1.0))
            normalized_power_hist.append(normalized_p)
        # Pad if needed
        while len(normalized_power_hist) < power_hist_len:
            normalized_power_hist.insert(0, 0.0)
        obs_list.extend(normalized_power_hist[:power_hist_len])
    
    # Temperature history
    if observation_config.get('enable_temperature_history', True):
        temp_hist = temperature_history or []
        normalized_temp_hist = []
        for t in temp_hist[-temp_hist_len:]:
            # Ensure t is a scalar
            t_val = float(t) if hasattr(t, '__float__') else float(t)
            normalized_t = float(np.clip((t_val - obs_temp_min) / (obs_temp_max - obs_temp_min), 0.0, 1.0))
            normalized_temp_hist.append(normalized_t)
        # Pad if needed
        while len(normalized_temp_hist) < temp_hist_len:
            normalized_temp_hist.insert(0, 0.0)
        obs_list.extend(normalized_temp_hist[:temp_hist_len])
    
    # Future toolpath
    if observation_config.get('enable_future_toolpath', False):
        if future_toolpath_len > 0:
            if toolpath_future is not None:
                # Use provided future toolpath
                # Handle list of arrays or single array
                if isinstance(toolpath_future, list) and len(toolpath_future) > 0:
                    # Convert list of arrays to 2D array
                    try:
                        future_path = np.array(toolpath_future[-future_toolpath_len:], dtype=np.float32)
                        # Ensure it's 2D: (n_points, 3)
                        if future_path.ndim == 1:
                            # If 1D, reshape to (1, 3) or pad
                            if len(future_path) == 3:
                                future_path = future_path.reshape(1, 3)
                            else:
                                # Pad to make it (n, 3)
                                n_points = len(future_path) // 3
                                future_path = future_path.reshape(n_points, 3)
                        elif future_path.ndim > 2:
                            # Flatten if needed
                            future_path = future_path.reshape(-1, 3)
                    except (ValueError, TypeError) as e:
                        # If conversion fails, create zeros
                        print(f"Warning: Could not convert toolpath_future to array: {e}")
                        future_path = np.zeros((future_toolpath_len, 3), dtype=np.float32)
                else:
                    future_path = np.zeros((future_toolpath_len, 3), dtype=np.float32)
            else:
                # Try to get from laser_all
                future_path = get_relative_toolpath(laser_all, future_toolpath_len, future=True)
            
            # Ensure we have the right shape (2D: n_points x 3)
            if future_path.ndim != 2:
                # Reshape to 2D if needed
                if future_path.ndim == 1:
                    # Assume it's flattened, reshape to (n, 3)
                    n_points = len(future_path) // 3
                    if n_points * 3 == len(future_path):
                        future_path = future_path.reshape(n_points, 3)
                    else:
                        # Pad if needed
                        padded = np.zeros((future_toolpath_len, 3), dtype=np.float32)
                        padded[:n_points, :] = future_path[:n_points*3].reshape(n_points, 3)
                        future_path = padded
                else:
                    # Flatten and reshape
                    future_path = future_path.flatten().reshape(-1, 3)
            
            # Ensure we have the right number of points
            if future_path.shape[0] < future_toolpath_len:
                # Pad if needed
                padding = np.zeros((future_toolpath_len - future_path.shape[0], 3), dtype=np.float32)
                future_path = np.concatenate([padding, future_path], axis=0)
            # Take only the requested length
            future_path = future_path[:future_toolpath_len]
            
            # Flatten and normalize (normalize by typical range, e.g., ±10 mm)
            future_path_flat = future_path.flatten()
            normalized = np.clip(future_path_flat / 10.0, -1.0, 1.0)  # Normalize to [-1, 1]
            normalized = (normalized + 1.0) / 2.0  # Convert to [0, 1]
            # Convert to list before extending to avoid dimension issues
            obs_list.extend(normalized.tolist())
    
    # History toolpath
    if observation_config.get('enable_history_toolpath', False):
        if history_toolpath_len > 0:
            if toolpath_history is not None:
                # Use provided history toolpath
                # Handle list of arrays or single array
                if isinstance(toolpath_history, list) and len(toolpath_history) > 0:
                    # Convert list of arrays to 2D array
                    try:
                        hist_path = np.array(toolpath_history[-history_toolpath_len:], dtype=np.float32)
                        # Ensure it's 2D: (n_points, 3)
                        if hist_path.ndim == 1:
                            # If 1D, reshape to (1, 3) or pad
                            if len(hist_path) == 3:
                                hist_path = hist_path.reshape(1, 3)
                            else:
                                # Pad to make it (n, 3)
                                n_points = len(hist_path) // 3
                                hist_path = hist_path.reshape(n_points, 3)
                        elif hist_path.ndim > 2:
                            # Flatten if needed
                            hist_path = hist_path.reshape(-1, 3)
                    except (ValueError, TypeError) as e:
                        # If conversion fails, create zeros
                        print(f"Warning: Could not convert toolpath_history to array: {e}")
                        hist_path = np.zeros((history_toolpath_len, 3), dtype=np.float32)
                else:
                    hist_path = np.zeros((history_toolpath_len, 3), dtype=np.float32)
            else:
                # Get from laser_all
                hist_path = get_relative_toolpath(laser_all, history_toolpath_len, future=False)
            
            # Ensure we have the right shape (2D: n_points x 3)
            if hist_path.ndim != 2:
                # Reshape to 2D if needed
                if hist_path.ndim == 1:
                    # Assume it's flattened, reshape to (n, 3)
                    n_points = len(hist_path) // 3
                    if n_points * 3 == len(hist_path):
                        hist_path = hist_path.reshape(n_points, 3)
                    else:
                        # Pad if needed
                        padded = np.zeros((history_toolpath_len, 3), dtype=np.float32)
                        padded[:n_points, :] = hist_path[:n_points*3].reshape(n_points, 3)
                        hist_path = padded
                else:
                    # Flatten and reshape
                    hist_path = hist_path.flatten().reshape(-1, 3)
            
            # Ensure we have the right number of points
            if hist_path.shape[0] < history_toolpath_len:
                # Pad if needed
                padding = np.zeros((history_toolpath_len - hist_path.shape[0], 3), dtype=np.float32)
                hist_path = np.concatenate([padding, hist_path], axis=0)
            # Take only the requested length
            hist_path = hist_path[:history_toolpath_len]
            
            # Flatten and normalize
            hist_path_flat = hist_path.flatten()
            normalized = np.clip(hist_path_flat / 10.0, -1.0, 1.0)  # Normalize to [-1, 1]
            normalized = (normalized + 1.0) / 2.0  # Convert to [0, 1]
            # Convert to list before extending to avoid dimension issues
            obs_list.extend(normalized.tolist())
            
            # Note: Power history for toolpath was removed (not included in observation)
    
    # Convert to numpy array
    # Ensure all elements are scalars (not arrays) to avoid dimension mismatch
    obs_list_flat = []
    for item in obs_list:
        if isinstance(item, (list, tuple, np.ndarray)):
            # If it's an array or list, flatten it and extend
            if isinstance(item, np.ndarray):
                obs_list_flat.extend(item.flatten().tolist())
            else:
                obs_list_flat.extend([float(x) for x in item])
        else:
            # It's a scalar, convert to float
            obs_list_flat.append(float(item))
    
    observation = np.array(obs_list_flat, dtype=np.float32)
    
    return observation


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
        self.history_length = params.get('observation_history_length', 10)
        self.temp_min = params.get('temp_min', 0.0)
        self.temp_max = params.get('temp_max', 5000.0)
        
        # Observation configuration
        self.observation_config = params.get('observation_config', {})
        
        # PID state variables
        self.integral_error = 0.0
        self.prev_error = 0.0
        self.prev_time = None
        
        # Store history
        self.observation_history = []
        self.action_history = []
        self.power_history = []
        self.error_history = []
        self.temperature_history = []  # Store temperature history for observations
        self.toolpath_history = []  # Store toolpath history
        self.power_history_toolpath = []  # Store power history for toolpath
    
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
    
    def _get_observation(
        self, 
        Levels: Union[List, Dict],
        laser_all: Optional[np.ndarray] = None,
        accum_time: Optional[np.ndarray] = None,
        accum_idx: Optional[int] = None,
        Properties: Optional[Dict] = None
    ) -> np.ndarray:
        """
        Get observation with history using class instance variables.
        
        Parameters:
        -----------
        Levels : list or dict
            List or dictionary containing all level data
        laser_all : np.ndarray, optional
            Array of laser positions
        accum_time : np.ndarray, optional
            Accumulated time array
        accum_idx : int, optional
            Index for local accumulated time
        Properties : dict, optional
            Properties dict containing T_liquidus
        
        Returns:
        --------
        np.ndarray
            Observation array with selected features
        """
        return get_observation(
            Levels,
            observation_config=self.observation_config,
            power_history=self.power_history,
            temperature_history=self.temperature_history,
            laser_all=laser_all,
            accum_time=accum_time,
            accum_idx=accum_idx,
            Properties=Properties,
            toolpath_history=self.toolpath_history,
            power_history_toolpath=self.power_history_toolpath,
            history_length=self.history_length,
            power_min=self.power_min,
            power_max=self.power_max,
            temp_min=self.temp_min,
            temp_max=self.temp_max
        )
    
    def step(self, Levels: Union[List, Dict], **kwargs) -> Tuple[float, bool]:
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
            # Get current temperature
            current_temp = get_max_temperature_level3(Levels)
            
            # Get full observation with history (for compatibility with RL)
            # Extract additional context from kwargs if provided
            laser_all = kwargs.get('laser_all', None)
            accum_time = kwargs.get('accum_time', None)
            accum_idx = kwargs.get('accum_idx', None)
            Properties = kwargs.get('Properties', None)
            observation = self._get_observation(Levels, laser_all, accum_time, accum_idx, Properties)
            
            # Calculate error (target - current)
            error = self.target_temperature - current_temp
            
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
            self.observation_history.append(observation)  # Store full observation array
            self.temperature_history.append(current_temp)  # Store temperature for next observation
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
        self.temperature_history.clear()
    
    def get_history(self) -> Dict:
        """Get control history for analysis."""
        return {
            'observations': self.observation_history.copy(),
            'actions': self.action_history.copy(),
            'power': self.power_history.copy(),
            'errors': self.error_history.copy(),
            'steps': list(range(0, len(self.observation_history) * self.update_interval, self.update_interval))
        }


class RLController:
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
        
        # Observation parameters
        self.history_length = self.controller_params.get('observation_history_length', 10)
        self.temp_min = self.controller_params.get('temp_min', 0.0)
        self.temp_max = self.controller_params.get('temp_max', 5000.0)
        self.power_min = self.controller_params.get('power_min', 0.0)
        self.power_max = self.controller_params.get('power_max', 500.0)
        
        # RL model (if provided)
        self.rl_model = None
        self.use_rl = False
        
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
        self.temperature_history = []  # Store temperature history for observations
        self.toolpath_history = []  # Store toolpath history
        self.power_history_toolpath = []  # Store power history for toolpath
        
        # Observation configuration
        self.observation_config = self.controller_params.get('observation_config', {})
        
        # Flag to print observation on first use
        self._print_first_observation = False
    
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
        power_adjustment = float(action[0]) * power_range * 0.90  # Scale to ±50% of range
        new_power = self.current_power + power_adjustment
        
        # Clamp to valid range
        new_power = np.clip(new_power, power_min, power_max)
        return float(new_power)
    
    def _get_observation(
        self, 
        Levels: Union[List, Dict],
        laser_all: Optional[np.ndarray] = None,
        accum_time: Optional[np.ndarray] = None,
        accum_idx: Optional[int] = None,
        Properties: Optional[Dict] = None
    ) -> np.ndarray:
        """
        Get observation with history using class instance variables.
        
        Parameters:
        -----------
        Levels : list or dict
            List or dictionary containing all level data
        laser_all : np.ndarray, optional
            Array of laser positions
        accum_time : np.ndarray, optional
            Accumulated time array
        accum_idx : int, optional
            Index for local accumulated time
        Properties : dict, optional
            Properties dict containing T_liquidus
        
        Returns:
        --------
        np.ndarray
            Observation array with selected features
        """
        return get_observation(
            Levels,
            observation_config=self.observation_config,
            power_history=self.power_history,
            temperature_history=self.temperature_history,
            laser_all=laser_all,
            accum_time=accum_time,
            accum_idx=accum_idx,
            Properties=Properties,
            toolpath_history=self.toolpath_history,
            power_history_toolpath=self.power_history_toolpath,
            history_length=self.history_length,
            power_min=self.power_min,
            power_max=self.power_max,
            temp_min=self.temp_min,
            temp_max=self.temp_max
        )
    
    def step(self, Levels: Union[List, Dict], **kwargs) -> Tuple[float, bool]:
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
            # Get current temperature for linear controller fallback
            current_temp = get_max_temperature_level3(Levels)
            
            # Get full observation with history
            # Extract kwargs for observation
            laser_all = kwargs.get('laser_all', None)
            accum_time = kwargs.get('accum_time', None)
            accum_idx = kwargs.get('accum_idx', None)
            Properties = kwargs.get('Properties', None)
            observation = self._get_observation(
                Levels,
                laser_all=laser_all,
                accum_time=accum_time,
                accum_idx=accum_idx,
                Properties=Properties
            )
            
            # Print observation breakdown on first use
            if self._print_first_observation:
                self.print_observation_breakdown(observation)
                self._print_first_observation = False
            
            # Compute action (new power)
            if self.use_rl and self.rl_model is not None:
                # Use RL agent with full observation
                action, _ = self.rl_model.predict(observation, deterministic=True)
                new_power = self._action_to_power(action)
            else:
                # Use linear controller (still uses single temperature value)
                new_power = compute_power_action(
                    observation=current_temp,
                    target_temperature=self.target_temperature,
                    current_power=self.current_power,
                    base_power=self.base_power,
                    controller_params=self.controller_params
                )
                print(f"Using linear controller with power: {new_power}")
            
            # Update current power
            self.current_power = new_power
            
            # Store history
            self.observation_history.append(observation)
            self.temperature_history.append(current_temp)  # Store temperature for next observation
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
        self.temperature_history.clear()
    
    def get_history(self) -> Dict:
        """Get control history for analysis/RL training."""
        return {
            'observations': self.observation_history.copy(),
            'actions': self.action_history.copy(),
            'power': self.power_history.copy(),
            'steps': list(range(0, len(self.observation_history) * self.update_interval, self.update_interval))
        }
    
    def print_observation_breakdown(self, observation: np.ndarray):
        """
        Print detailed breakdown of observation array.
        
        Parameters:
        -----------
        observation : np.ndarray
            Observation array to print
        """
        from observation_utils import print_observation_breakdown as print_obs
        
        obs_config = self.observation_config if hasattr(self, 'observation_config') and self.observation_config else {}
        
        print()  # Add newline before breakdown
        print_obs(
            observation=observation,
            observation_config=obs_config,
            history_length=self.history_length,
            power_min=self.power_min,
            power_max=self.power_max,
            temp_min=self.temp_min,
            temp_max=self.temp_max,
            prefix="  "
        )

