"""
Observation Utilities for GO-MELT RL Training

This module provides shared utilities for building and analyzing observation arrays
used in both training and runtime environments.
"""

import numpy as np
from typing import Dict, Optional, List, Any, Tuple
import sys
import os

# Import controller functions for observation features
try:
    from controller import (
        get_max_temperature_level3,
        get_fraction_above_liquidus,
        get_accum_time_local,
        get_laser_speed,
        get_distance_to_last_point
    )
    HAS_CONTROLLER = True
except ImportError:
    HAS_CONTROLLER = False
    get_max_temperature_level3 = None
    get_fraction_above_liquidus = None
    get_accum_time_local = None
    get_laser_speed = None
    get_distance_to_last_point = None

# Try to import JAX for handling JAX arrays
try:
    import jax.numpy as jnp
    HAS_JAX = True
except ImportError:
    HAS_JAX = False
    jnp = None


def calculate_observation_size(
    observation_config: Optional[Dict] = None,
    default_history_length: int = 10
) -> int:
    """
    Calculate the size of observation array based on observation_config.
    
    Parameters:
    -----------
    observation_config : dict, optional
        Configuration dict specifying which features are enabled
    default_history_length : int
        Default history length if not specified in config. Default: 10
        
    Returns:
    --------
    int
        Size of observation array
    """
    size = 0
    
    if not observation_config:
        # Simple observation: current temp + power history + temp history
        return 1 + 2 * default_history_length
    
    # Temperature statistics (only max is supported)
    if observation_config.get('enable_max_T_L3', True):
        size += 1
    
    # Fraction above liquidus
    if observation_config.get('enable_frac_above_liquidus', False):
        size += 1
    
    # Accumulated time local
    if observation_config.get('enable_accum_time_local', False):
        size += 1
    
    # Laser speed
    if observation_config.get('enable_laser_speed', False):
        size += 1
    
    # Distance to last point
    if observation_config.get('enable_distance_to_last_point', False):
        size += 1
    
    # Power history
    if observation_config.get('enable_power_history', True):
        power_hist_len = observation_config.get('power_history_length', default_history_length)
        size += power_hist_len
    
    # Temperature history
    if observation_config.get('enable_temperature_history', True):
        temp_hist_len = observation_config.get('temperature_history_length', default_history_length)
        size += temp_hist_len
    
    # Future toolpath
    if observation_config.get('enable_future_toolpath', False):
        future_toolpath_len = observation_config.get('future_toolpath_length', 10)
        size += future_toolpath_len * 3  # x, y, z for each point
    
    # History toolpath
    if observation_config.get('enable_history_toolpath', False):
        history_toolpath_len = observation_config.get('history_toolpath_length', 10)
        size += history_toolpath_len * 3  # x, y, z for each point
    
    # If size is 0, fall back to simple observation
    if size == 0:
        size = 1 + 2 * default_history_length
    
    return size


def build_observation(
    observation_config: Optional[Dict],
    temperature: float,
    power: float,
    power_history: List[float],
    temperature_history: List[float],
    levels: Optional[Any] = None,
    properties: Optional[Dict] = None,
    laser_all: Optional[np.ndarray] = None,
    accum_time: Optional[np.ndarray] = None,
    accum_idx: Optional[int] = None,
    future_toolpath: Optional[np.ndarray] = None,
    history_toolpath: Optional[np.ndarray] = None,
    power_min: float = 0.0,
    power_max: float = 500.0,
    temp_min: float = 0.0,
    temp_max: float = 5000.0,
    default_history_length: int = 10,
    toolpath_normalization_range: float = 10.0
) -> np.ndarray:
    """
    Build observation array based on observation_config.
    
    This function constructs an observation array with features specified in
    observation_config. All features from rl.json (lines 145-158) are supported.
    
    Parameters:
    -----------
    observation_config : dict, optional
        Configuration dict specifying which features are enabled. If None, uses simple format.
    temperature : float
        Current maximum temperature in Level 3 (Kelvin)
    power : float
        Current power (Watts)
    power_history : list
        List of previous power values
    temperature_history : list
        List of previous temperature values
    levels : list or dict, optional
        GO-MELT Levels structure (for accessing temperature fields and stats)
    properties : dict, optional
        Properties dict (for T_liquidus, etc.)
    laser_all : np.ndarray, optional
        Array of laser positions for speed/distance calculations
    accum_time : np.ndarray, optional
        Accumulated time array for local time calculations
    accum_idx : int, optional
        Index for accumulated time local calculation
    future_toolpath : np.ndarray, optional
        Array of future toolpath points (shape: [N, 3])
    history_toolpath : np.ndarray, optional
        Array of history toolpath points (shape: [N, 3])
    power_min : float
        Minimum power for normalization. Default: 0.0
    power_max : float
        Maximum power for normalization. Default: 500.0
    temp_min : float
        Minimum temperature for normalization. Default: 0.0
    temp_max : float
        Maximum temperature for normalization. Default: 5000.0
    default_history_length : int
        Default history length if not specified in config. Default: 10
    toolpath_normalization_range : float
        Range for toolpath normalization (relative positions in mm). Default: 10.0
        
    Returns:
    --------
    np.ndarray
        Observation array with features specified in observation_config
    """
    obs_list = []
    
    # Helper functions for normalization
    def normalize_temperature(temp: float) -> float:
        return np.clip((temp - temp_min) / (temp_max - temp_min), 0.0, 1.0)
    
    def normalize_power(pwr: float) -> float:
        return np.clip((pwr - power_min) / (power_max - power_min), 0.0, 1.0)
    
    # If no config, use simple format
    if not observation_config:
        norm_temp = normalize_temperature(temperature)
        power_hist = power_history[-default_history_length:]
        temp_hist = temperature_history[-default_history_length:]
        
        norm_power_hist = [normalize_power(p) for p in power_hist]
        norm_temp_hist = [normalize_temperature(t) for t in temp_hist]
        
        while len(norm_power_hist) < default_history_length:
            norm_power_hist.insert(0, 0.0)
        while len(norm_temp_hist) < default_history_length:
            norm_temp_hist.insert(0, 0.0)
        
        obs_list = [norm_temp] + norm_power_hist + norm_temp_hist
        return np.array(obs_list, dtype=np.float32)
    
    # Temperature statistics (only max is supported)
    if observation_config.get('enable_max_T_L3', True):
        norm_temp = normalize_temperature(temperature)
        obs_list.append(norm_temp)
    
    # Fraction above liquidus
    if observation_config.get('enable_frac_above_liquidus', False):
        if properties and 'T_liquidus' in properties and levels is not None:
            T_liquidus = properties['T_liquidus']
            if get_fraction_above_liquidus is not None:
                try:
                    frac = float(get_fraction_above_liquidus(levels, T_liquidus))
                except:
                    frac = 0.0
            else:
                # Fallback: calculate manually
                try:
                    temp_field = levels[3]["T0"]
                    if hasattr(temp_field, 'block_until_ready'):
                        temp_field = np.array(temp_field)
                    elif hasattr(temp_field, '__array__'):
                        temp_field = np.asarray(temp_field)
                    above_liquidus = np.sum(temp_field > T_liquidus)
                    total_cells = len(temp_field)
                    frac = float(above_liquidus / total_cells) if total_cells > 0 else 0.0
                except:
                    frac = 0.0
        else:
            frac = 0.0
        obs_list.append(frac)
    
    # Accumulated time local
    if observation_config.get('enable_accum_time_local', False):
        if get_accum_time_local is not None and accum_time is not None:
            try:
                accum_local = get_accum_time_local(accum_time, accum_idx)
                # Normalize by reasonable max (e.g., 1 second)
                normalized = float(np.clip(accum_local / 1.0, 0.0, 1.0))
            except:
                normalized = 0.0
        else:
            normalized = 0.0
        obs_list.append(normalized)
    
    # Laser speed
    if observation_config.get('enable_laser_speed', False):
        if get_laser_speed is not None and laser_all is not None:
            try:
                speed = get_laser_speed(laser_all)
                # Normalize by typical max speed (e.g., 2000 mm/s)
                normalized = float(np.clip(speed / 2000.0, 0.0, 1.0))
            except:
                normalized = 0.0
        else:
            normalized = 0.0
        obs_list.append(normalized)
    
    # Distance to last point
    if observation_config.get('enable_distance_to_last_point', False):
        if get_distance_to_last_point is not None and laser_all is not None:
            try:
                distance = get_distance_to_last_point(laser_all)
                # Normalize by typical max distance (e.g., 1 mm)
                normalized = float(np.clip(distance / 1.0, 0.0, 1.0))
            except:
                normalized = 0.0
        else:
            normalized = 0.0
        obs_list.append(normalized)
    
    # Power history
    if observation_config.get('enable_power_history', True):
        power_hist_len = observation_config.get('power_history_length', default_history_length)
        power_hist = power_history[-power_hist_len:]
        norm_power_hist = [normalize_power(p) for p in power_hist]
        while len(norm_power_hist) < power_hist_len:
            norm_power_hist.insert(0, 0.0)
        obs_list.extend(norm_power_hist[:power_hist_len])
    
    # Temperature history
    if observation_config.get('enable_temperature_history', True):
        temp_hist_len = observation_config.get('temperature_history_length', default_history_length)
        temp_hist = temperature_history[-temp_hist_len:]
        norm_temp_hist = [normalize_temperature(t) for t in temp_hist]
        while len(norm_temp_hist) < temp_hist_len:
            norm_temp_hist.insert(0, 0.0)
        obs_list.extend(norm_temp_hist[:temp_hist_len])
    
    # Future toolpath (relative positions, normalized)
    if observation_config.get('enable_future_toolpath', False):
        future_toolpath_len = observation_config.get('future_toolpath_length', 10)
        if future_toolpath is not None and len(future_toolpath) > 0:
            # Take first N points
            future_path = future_toolpath[:min(future_toolpath_len, len(future_toolpath))]
            # Pad if needed
            if len(future_path) < future_toolpath_len:
                padding = np.zeros((future_toolpath_len - len(future_path), 3), dtype=np.float32)
                future_path = np.vstack([future_path, padding])
            # Normalize relative path within config-defined range
            toolpath_range = observation_config.get('toolpath_normalization_range', toolpath_normalization_range)
            future_path_flat = future_path.flatten()
            # Normalize: divide by range, clip to [-1, 1], then convert to [0, 1]
            normalized = np.clip(future_path_flat / toolpath_range, -1.0, 1.0)
            normalized = (normalized + 1.0) / 2.0  # Convert to [0, 1]
            obs_list.extend(normalized.tolist())
        else:
            # Pad with zeros if not available
            obs_list.extend([0.0] * (future_toolpath_len * 3))
    
    # History toolpath (relative positions, normalized)
    if observation_config.get('enable_history_toolpath', False):
        history_toolpath_len = observation_config.get('history_toolpath_length', 10)
        if history_toolpath is not None and len(history_toolpath) > 0:
            # Take first N points
            hist_path = history_toolpath[:min(history_toolpath_len, len(history_toolpath))]
            # Pad if needed
            if len(hist_path) < history_toolpath_len:
                padding = np.zeros((history_toolpath_len - len(hist_path), 3), dtype=np.float32)
                hist_path = np.vstack([hist_path, padding])
            # Normalize relative path within config-defined range
            toolpath_range = observation_config.get('toolpath_normalization_range', toolpath_normalization_range)
            hist_path_flat = hist_path.flatten()
            # Normalize: divide by range, clip to [-1, 1], then convert to [0, 1]
            normalized = np.clip(hist_path_flat / toolpath_range, -1.0, 1.0)
            normalized = (normalized + 1.0) / 2.0  # Convert to [0, 1]
            obs_list.extend(normalized.tolist())
        else:
            # Pad with zeros if not available
            obs_list.extend([0.0] * (history_toolpath_len * 3))
    
    # If no features were added, fall back to simple format
    if not obs_list:
        norm_temp = normalize_temperature(temperature)
        power_hist = power_history[-default_history_length:]
        temp_hist = temperature_history[-default_history_length:]
        
        norm_power_hist = [normalize_power(p) for p in power_hist]
        norm_temp_hist = [normalize_temperature(t) for t in temp_hist]
        
        while len(norm_power_hist) < default_history_length:
            norm_power_hist.insert(0, 0.0)
        while len(norm_temp_hist) < default_history_length:
            norm_temp_hist.insert(0, 0.0)
        
        obs_list = [norm_temp] + norm_power_hist + norm_temp_hist
    
    return np.array(obs_list, dtype=np.float32)


def print_observation_breakdown(
    observation: np.ndarray,
    observation_config: Optional[Dict] = None,
    history_length: int = 10,
    power_min: float = 0.0,
    power_max: float = 500.0,
    temp_min: float = 0.0,
    temp_max: float = 5000.0,
    toolpath_normalization_range: float = 10.0,
    prefix: str = "  "
) -> None:
    """
    Print detailed breakdown of observation array.
    
    This function analyzes an observation array and prints what each element represents
    based on the observation_config. It works with both training environments and
    runtime controllers.
    
    Parameters:
    -----------
    observation : np.ndarray
        Observation array to print
    observation_config : dict, optional
        Configuration dict specifying which features are enabled. If None, uses simple format.
    history_length : int
        Default history length if not specified in config. Default: 10
    power_min : float
        Minimum power for denormalization. Default: 0.0
    power_max : float
        Maximum power for denormalization. Default: 500.0
    temp_min : float
        Minimum temperature for denormalization. Default: 0.0
    temp_max : float
        Maximum temperature for denormalization. Default: 5000.0
    toolpath_normalization_range : float
        Range for toolpath normalization (relative positions in mm). Default: 10.0
    prefix : str
        Prefix string for each printed line (for indentation). Default: "  "
    """
    obs_flat = observation.flatten()
    obs_config = observation_config if observation_config else {}
    
    print(f"{prefix}RL Observation Breakdown (size: {len(obs_flat)}):")
    
    # If no config, use simple format
    if not obs_config:
        print(f"{prefix}  Simple format: [current_temp, power_history, temp_history]")
        if len(obs_flat) > 0:
            temp_norm = obs_flat[0]
            temp_actual = temp_norm * (temp_max - temp_min) + temp_min
            print(f"{prefix}  [0] Current Temperature: {temp_norm:.6f} -> {temp_actual:.2f}K")
        return
    
    # Build observation in the same order as build_observation function
    idx = 0
    
    # Temperature statistics (only max is supported)
    if obs_config.get('enable_max_T_L3', True):
        if idx < len(obs_flat):
            temp_norm = obs_flat[idx]
            temp_actual = temp_norm * (temp_max - temp_min) + temp_min
            print(f"{prefix}  [{idx}] Max Temperature L3: {temp_norm:.6f} -> {temp_actual:.2f}K")
            idx += 1
    
    # Fraction above liquidus
    if obs_config.get('enable_frac_above_liquidus', False):
        if idx < len(obs_flat):
            frac = obs_flat[idx]
            print(f"{prefix}  [{idx}] Fraction above liquidus: {frac:.6f}")
            idx += 1
    
    # Accumulated time local
    if obs_config.get('enable_accum_time_local', False):
        if idx < len(obs_flat):
            print(f"{prefix}  [{idx}] Accumulated time local: {obs_flat[idx]:.6f}")
            idx += 1
    
    # Laser speed
    if obs_config.get('enable_laser_speed', False):
        if idx < len(obs_flat):
            print(f"{prefix}  [{idx}] Laser speed: {obs_flat[idx]:.6f}")
            idx += 1
    
    # Distance to last point (only if enabled in config)
    if obs_config.get('enable_distance_to_last_point', False):
        if idx < len(obs_flat):
            print(f"{prefix}  [{idx}] Distance to last point: {obs_flat[idx]:.6f}")
            idx += 1
    
    # Note: Current power is NOT included (removed from observation)
    
    # Power history
    if obs_config.get('enable_power_history', True):
        power_hist_len = obs_config.get('power_history_length', history_length)
        if idx + power_hist_len <= len(obs_flat):
            print(f"{prefix}  Power History (indices {idx}-{idx+power_hist_len-1}):")
            for i in range(min(3, power_hist_len)):  # Show first 3
                if idx + i < len(obs_flat):
                    power_norm = obs_flat[idx + i]
                    if power_norm > 0:
                        power_actual = power_norm * (power_max - power_min) + power_min
                        print(f"{prefix}    [{idx + i}] Power[{i}]: {power_norm:.6f} -> {power_actual:.2f}W")
                    else:
                        print(f"{prefix}    [{idx + i}] Power[{i}]: {power_norm:.6f} -> (padded/zero)")
            if power_hist_len > 3:
                print(f"{prefix}    ... ({power_hist_len - 3} more power history values)")
            idx += power_hist_len
    
    # Temperature history
    if obs_config.get('enable_temperature_history', True):
        temp_hist_len = obs_config.get('temperature_history_length', history_length)
        if idx + temp_hist_len <= len(obs_flat):
            print(f"{prefix}  Temperature History (indices {idx}-{idx+temp_hist_len-1}):")
            for i in range(min(3, temp_hist_len)):  # Show first 3
                if idx + i < len(obs_flat):
                    temp_norm = obs_flat[idx + i]
                    if temp_norm > 0:
                        temp_actual = temp_norm * (temp_max - temp_min) + temp_min
                        print(f"{prefix}    [{idx + i}] Temp[{i}]: {temp_norm:.6f} -> {temp_actual:.2f}K")
                    else:
                        print(f"{prefix}    [{idx + i}] Temp[{i}]: {temp_norm:.6f} -> (padded/zero)")
            if temp_hist_len > 3:
                print(f"{prefix}    ... ({temp_hist_len - 3} more temperature history values)")
            idx += temp_hist_len
    
    # Future toolpath
    if obs_config.get('enable_future_toolpath', False):
        future_toolpath_len = obs_config.get('future_toolpath_length', 10)
        expected_size = future_toolpath_len * 3
        actual_size = min(expected_size, len(obs_flat) - idx)
        if actual_size > 0:
            num_points = actual_size // 3
            print(f"{prefix}  Future Toolpath (indices {idx}-{idx+actual_size-1}): {num_points} points (x,y,z each, actual: {actual_size} values)")
            for i in range(num_points):
                x_idx = idx + i * 3
                y_idx = idx + i * 3 + 1
                z_idx = idx + i * 3 + 2
                if x_idx < len(obs_flat) and y_idx < len(obs_flat) and z_idx < len(obs_flat):
                    x_norm = obs_flat[x_idx]
                    y_norm = obs_flat[y_idx]
                    z_norm = obs_flat[z_idx]
                    # Denormalize: (normalized * 2.0 - 1.0) * range = relative position in mm
                    # Get range from config if available, otherwise use default
                    toolpath_range = obs_config.get('toolpath_normalization_range', toolpath_normalization_range)
                    x_actual = (x_norm * 2.0 - 1.0) * toolpath_range
                    y_actual = (y_norm * 2.0 - 1.0) * toolpath_range
                    z_actual = (z_norm * 2.0 - 1.0) * toolpath_range
                    print(f"{prefix}    Point[{i}]: x={x_norm:.6f} -> {x_actual:.3f}mm, y={y_norm:.6f} -> {y_actual:.3f}mm, z={z_norm:.6f} -> {z_actual:.3f}mm")
            idx += actual_size

        else:
            print(f"{prefix}  Future Toolpath: Not available (expected {expected_size} values)")
    
    # History toolpath
    if obs_config.get('enable_history_toolpath', False):
        history_toolpath_len = obs_config.get('history_toolpath_length', 10)
        expected_size = history_toolpath_len * 3
        actual_size = min(expected_size, len(obs_flat) - idx)
        if actual_size > 0:
            num_points = actual_size // 3
            print(f"{prefix}  History Toolpath (indices {idx}-{idx+actual_size-1}): {num_points} points (x,y,z each, actual: {actual_size} values)")
            for i in range(num_points):
                x_idx = idx + i * 3
                y_idx = idx + i * 3 + 1
                z_idx = idx + i * 3 + 2
                if x_idx < len(obs_flat) and y_idx < len(obs_flat) and z_idx < len(obs_flat):
                    x_norm = obs_flat[x_idx]
                    y_norm = obs_flat[y_idx]
                    z_norm = obs_flat[z_idx]
                    # Denormalize: (normalized * 2.0 - 1.0) * range = relative position in mm
                    # Get range from config if available, otherwise use default
                    toolpath_range = obs_config.get('toolpath_normalization_range', toolpath_normalization_range)
                    x_actual = (x_norm * 2.0 - 1.0) * toolpath_range
                    y_actual = (y_norm * 2.0 - 1.0) * toolpath_range
                    z_actual = (z_norm * 2.0 - 1.0) * toolpath_range
                    print(f"{prefix}    Point[{i}]: x={x_norm:.6f} -> {x_actual:.3f}mm, y={y_norm:.6f} -> {y_actual:.3f}mm, z={z_norm:.6f} -> {z_actual:.3f}mm")
            idx += actual_size
        else:
            print(f"{prefix}  History Toolpath: Not available (expected {expected_size} values)")
    
    # Check if there are remaining values
    if idx < len(obs_flat):
        remaining = len(obs_flat) - idx
        print(f"{prefix}  WARNING: {remaining} unexpected values at end of observation (indices {idx}-{len(obs_flat)-1})")
        print(f"{prefix}  This suggests a mismatch between config and actual observation building.")
    
    print(f"{prefix}\n{prefix}  Total observation size: {len(obs_flat)}")
    expected_str = str(idx) if idx == len(obs_flat) else 'MISMATCH'
    print(f"{prefix}  Expected size based on config: {expected_str}")
