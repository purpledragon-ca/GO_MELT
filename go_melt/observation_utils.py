"""
Observation Utilities for GO-MELT RL Training

This module provides shared utilities for analyzing and printing observation arrays
used in both training and runtime environments.
"""

import numpy as np
from typing import Dict, Optional


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
    
    # Build observation in the same order as get_observation function
    idx = 0
    
    # Temperature statistics
    if obs_config.get('enable_max_T_L3', True):
        if idx < len(obs_flat):
            temp_norm = obs_flat[idx]
            temp_actual = temp_norm * (temp_max - temp_min) + temp_min
            print(f"{prefix}  [{idx}] Max Temperature L3: {temp_norm:.6f} -> {temp_actual:.2f}K")
            idx += 1
    
    if obs_config.get('enable_avg_T_L3', False):
        if idx < len(obs_flat):
            temp_norm = obs_flat[idx]
            temp_actual = temp_norm * (temp_max - temp_min) + temp_min
            print(f"{prefix}  [{idx}] Avg Temperature L3: {temp_norm:.6f} -> {temp_actual:.2f}K")
            idx += 1
    
    if obs_config.get('enable_min_T_L3', False):
        if idx < len(obs_flat):
            temp_norm = obs_flat[idx]
            temp_actual = temp_norm * (temp_max - temp_min) + temp_min
            print(f"{prefix}  [{idx}] Min Temperature L3: {temp_norm:.6f} -> {temp_actual:.2f}K")
            idx += 1
    
    if obs_config.get('enable_std_T_L3', False):
        if idx < len(obs_flat):
            std_norm = obs_flat[idx]
            std_actual = std_norm * (temp_max - temp_min)
            print(f"{prefix}  [{idx}] Std Temperature L3: {std_norm:.6f} -> {std_actual:.2f}K")
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

