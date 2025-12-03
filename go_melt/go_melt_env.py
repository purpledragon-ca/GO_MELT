"""
Gym-like Environment Wrapper for GO-MELT Simulation

This module provides a Gym-compatible environment interface for training
RL agents to control laser power in GO-MELT simulations.

Note: This is a simplified environment for training. For production use,
the RL agent should be integrated directly into the RLController class.
"""

import os
import sys
import json
import numpy as np
from typing import Dict, Tuple, Optional, Any
from pathlib import Path
import gymnasium as gym
from gymnasium import spaces

# Add parent directory to path for imports
_current_dir = os.path.dirname(os.path.abspath(__file__))
if _current_dir not in sys.path:
    sys.path.insert(0, _current_dir)

try:
    from controller import get_max_temperature_level3, RLController, get_observation as get_full_observation
except ImportError:
    RLController = None
    get_max_temperature_level3 = None
    get_full_observation = None


class GoMeltEnv(gym.Env):
    """
    Gym environment wrapper for GO-MELT simulation.
    
    The environment provides:
    - Observation: Array containing:
        - Current maximum temperature in Level 3 (normalized)
        - Power history (past N iterations, normalized)
        - Temperature history (past N iterations, normalized)
      Shape: (1 + 2*observation_history_length,)
    - Action: Power adjustment (continuous, normalized to [-1, 1])
    - Reward: Negative squared error from target temperature
    """
    
    metadata = {"render_modes": ["human"], "render_fps": 4}
    
    def __init__(
        self,
        config_file: str,
        target_temperature: float = 3000.0,
        power_min: float = 0.0,
        power_max: float = 500.0,
        base_power: float = 285.0,
        update_interval: int = 10,
        max_steps: Optional[int] = None,
        reward_scale: float = 0.1,
        render_mode: Optional[str] = None,
        device_id: int = 0,
        observation_history_length: int = 10
    ):
        """
        Initialize the GO-MELT environment.
        
        Parameters:
        -----------
        config_file : str
            Path to JSON configuration file for GO-MELT
        target_temperature : float
            Target maximum temperature (Kelvin). Default: 3000.0
        power_min : float
            Minimum allowed power (Watts). Default: 0.0
        power_max : float
            Maximum allowed power (Watts). Default: 500.0
        base_power : float
            Initial/base power level (Watts). Default: 285.0
        update_interval : int
            Number of simulation steps between power updates. Default: 10
        max_steps : int, optional
            Maximum number of steps per episode. If None, runs until simulation ends.
        reward_scale : float
            Scaling factor for rewards. Default: 1.0
        render_mode : str, optional
            Rendering mode. Currently not implemented.
        device_id : int
            GPU device ID. Default: 0
        observation_history_length : int
            Number of history steps to include in observation. Default: 10
        """
        super().__init__()
        
        self.config_file = config_file
        self.target_temperature = target_temperature
        self.power_min = power_min
        self.power_max = power_max
        self.base_power = base_power
        self.update_interval = update_interval
        self.max_steps = max_steps
        self.reward_scale = reward_scale
        self.render_mode = render_mode
        self.device_id = device_id
        
        # Load configuration
        with open(config_file, 'r') as f:
            self.solver_input = json.load(f)
        
        # Read observation configuration from config if available
        if "power_controller" in self.solver_input:
            controller_params = self.solver_input["power_controller"].get("controller_params", {})
            if "observation_history_length" in controller_params:
                self.observation_history_length = controller_params["observation_history_length"]
            else:
                self.observation_history_length = observation_history_length
            
            # Get observation_config if available
            self.observation_config = controller_params.get("observation_config", {})
        else:
            self.observation_history_length = observation_history_length
            self.observation_config = {}
        
        # Ensure RL controller is configured
        if "power_controller" not in self.solver_input:
            self.solver_input["power_controller"] = {}
        
        self.solver_input["power_controller"]["type"] = "rl"
        self.solver_input["power_controller"]["target_temperature"] = target_temperature
        self.solver_input["power_controller"]["update_interval"] = update_interval
        if "controller_params" not in self.solver_input["power_controller"]:
            self.solver_input["power_controller"]["controller_params"] = {}
        self.solver_input["power_controller"]["controller_params"].update({
            "power_min": power_min,
            "power_max": power_max,
            "observation_history_length": self.observation_history_length,
            "observation_config": self.observation_config
        })
        
        # Store Properties for observation (needed for some features like frac_above_liquidus)
        self.properties = self.solver_input.get("properties", {})
        
        # Environment state
        self.current_power = base_power
        self.step_count = 0
        self.episode_reward = 0.0
        self.episode_length = 0
        self.simulation_complete = False
        
        # For storing simulation state
        self.levels = None
        self.power_controller = None
        
        # History tracking for observations
        self.power_history = []
        self.temperature_history = []
        self.toolpath_history = []  # For toolpath history if enabled
        self.toolpath_future = []  # For future toolpath if enabled
        self.power_history_toolpath = []  # Power history for toolpath
        
        # Define action and observation spaces
        # Action: normalized power adjustment [-1, 1] -> maps to power range
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(1,),
            dtype=np.float32
        )
        
        # Using temperature range 0-5000K and power range for normalization
        self.temp_min = 0.0
        self.temp_max = 5000.0
        self.power_min = power_min
        self.power_max = power_max
        
        # Calculate observation space size based on config
        # We'll create a test observation to determine the size
        obs_size = self._calculate_observation_size()
        
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(obs_size,),
            dtype=np.float32
        )
    
    def _normalize_temperature(self, temp: float) -> float:
        """Normalize temperature to [0, 1] range."""
        return np.clip((temp - self.temp_min) / (self.temp_max - self.temp_min), 0.0, 1.0)
    
    def _denormalize_temperature(self, norm_temp: float) -> float:
        """Denormalize temperature from [0, 1] range."""
        return norm_temp * (self.temp_max - self.temp_min) + self.temp_min
    
    def _normalize_power(self, power: float) -> float:
        """Normalize power to [0, 1] range."""
        return np.clip((power - self.power_min) / (self.power_max - self.power_min), 0.0, 1.0)
    
    def _denormalize_power(self, norm_power: float) -> float:
        """Denormalize power from [0, 1] range."""
        return norm_power * (self.power_max - self.power_min) + self.power_min
    
    def _create_mock_levels(self, temperature: float) -> Dict:
        """
        Create a mock Levels structure for use with get_observation.
        
        This is needed because the simplified training environment doesn't run
        actual simulations, but the full observation function expects Levels data.
        """
        # Create a simple mock temperature field
        # For training, we just need the max temperature, so we create a minimal structure
        mock_temp_field = np.array([[temperature]], dtype=np.float32)
        
        return {
            3: {
                "T0": mock_temp_field
            }
        }
    
    def _calculate_observation_size(self) -> int:
        """
        Calculate the observation size based on the observation_config.
        """
        size = 0
        
        # If we have observation_config, calculate size based on enabled features
        if self.observation_config:
            # Temperature statistics
            if self.observation_config.get('enable_max_T_L3', True):
                size += 1
            if self.observation_config.get('enable_avg_T_L3', False):
                size += 1
            if self.observation_config.get('enable_min_T_L3', False):
                size += 1
            if self.observation_config.get('enable_std_T_L3', False):
                size += 1
            
            # Fraction above liquidus
            if self.observation_config.get('enable_frac_above_liquidus', False):
                size += 1
            
            # Accumulated time local
            if self.observation_config.get('enable_accum_time_local', False):
                size += 1
            
            # Laser speed
            if self.observation_config.get('enable_laser_speed', False):
                size += 1
            
            # Distance to last point
            if self.observation_config.get('enable_distance_to_last_point', False):
                size += 1
            
            # Power history
            if self.observation_config.get('enable_power_history', True):
                power_hist_len = self.observation_config.get('power_history_length', self.observation_history_length)
                size += power_hist_len
            
            # Temperature history
            if self.observation_config.get('enable_temperature_history', True):
                temp_hist_len = self.observation_config.get('temperature_history_length', self.observation_history_length)
                size += temp_hist_len
            
            # Future toolpath
            if self.observation_config.get('enable_future_toolpath', False):
                future_toolpath_len = self.observation_config.get('future_toolpath_length', 10)
                size += future_toolpath_len * 3  # x, y, z for each point
            
            # History toolpath
            if self.observation_config.get('enable_history_toolpath', False):
                history_toolpath_len = self.observation_config.get('history_toolpath_length', 10)
                size += history_toolpath_len * 3  # x, y, z for each point
        
        # If no config or size is 0, fall back to simple observation
        if size == 0:
            # Simple observation: current temp + power history + temp history
            size = 1 + 2 * self.observation_history_length
        
        return size
    
    def _get_observation(self, temperature: float, power: float) -> np.ndarray:
        """
        Get observation array with configurable features.
        
        Builds observation based on observation_config if provided,
        otherwise falls back to simple observation format.
        
        Parameters:
        -----------
        temperature : float
            Current temperature
        power : float
            Current power
            
        Returns:
        --------
        np.ndarray
            Observation array with features specified in observation_config
        """
        obs_list = []
        
        # If we have observation_config, build observation based on enabled features
        if self.observation_config:
            # Temperature statistics
            if self.observation_config.get('enable_max_T_L3', True):
                norm_temp = self._normalize_temperature(temperature)
                obs_list.append(norm_temp)
            
            if self.observation_config.get('enable_avg_T_L3', False):
                # For simplified env, use max temp as approximation
                norm_temp = self._normalize_temperature(temperature)
                obs_list.append(norm_temp)
            
            if self.observation_config.get('enable_min_T_L3', False):
                # For simplified env, use a lower estimate
                min_temp_est = temperature * 0.8  # Rough estimate
                norm_temp = self._normalize_temperature(min_temp_est)
                obs_list.append(norm_temp)
            
            if self.observation_config.get('enable_std_T_L3', False):
                # For simplified env, estimate std (small value)
                std_est = temperature * 0.05  # Rough estimate
                norm_std = np.clip(std_est / (self.temp_max - self.temp_min), 0.0, 1.0)
                obs_list.append(norm_std)
            
            # Fraction above liquidus
            if self.observation_config.get('enable_frac_above_liquidus', False):
                if self.properties and 'T_liquidus' in self.properties:
                    T_liquidus = self.properties['T_liquidus']
                    frac = 1.0 if temperature > T_liquidus else 0.0
                else:
                    frac = 0.0
                obs_list.append(frac)
            
            # Accumulated time local (not available in simplified env)
            if self.observation_config.get('enable_accum_time_local', False):
                obs_list.append(0.0)  # Not available in simplified env
            
            # Laser speed (not available in simplified env)
            if self.observation_config.get('enable_laser_speed', False):
                obs_list.append(0.0)  # Not available in simplified env
            
            # Distance to last point (not available in simplified env)
            if self.observation_config.get('enable_distance_to_last_point', False):
                obs_list.append(0.0)  # Not available in simplified env
            
            # Power history
            if self.observation_config.get('enable_power_history', True):
                power_hist_len = self.observation_config.get('power_history_length', self.observation_history_length)
                power_hist = self.power_history[-power_hist_len:]
                norm_power_hist = [self._normalize_power(p) for p in power_hist]
                while len(norm_power_hist) < power_hist_len:
                    norm_power_hist.insert(0, 0.0)
                obs_list.extend(norm_power_hist[:power_hist_len])
            
            # Temperature history
            if self.observation_config.get('enable_temperature_history', True):
                temp_hist_len = self.observation_config.get('temperature_history_length', self.observation_history_length)
                temp_hist = self.temperature_history[-temp_hist_len:]
                norm_temp_hist = [self._normalize_temperature(t) for t in temp_hist]
                while len(norm_temp_hist) < temp_hist_len:
                    norm_temp_hist.insert(0, 0.0)
                obs_list.extend(norm_temp_hist[:temp_hist_len])
            
            # Future toolpath (not available in simplified env)
            if self.observation_config.get('enable_future_toolpath', False):
                future_toolpath_len = self.observation_config.get('future_toolpath_length', 10)
                # Pad with zeros (3 values per point: x, y, z)
                obs_list.extend([0.0] * (future_toolpath_len * 3))
            
            # History toolpath (not available in simplified env)
            if self.observation_config.get('enable_history_toolpath', False):
                history_toolpath_len = self.observation_config.get('history_toolpath_length', 10)
                # Pad with zeros (3 values per point: x, y, z)
                obs_list.extend([0.0] * (history_toolpath_len * 3))
        
        # If no config or empty list, fall back to simple observation format
        if not obs_list:
            # Simple observation: current temp + power history + temp history
            norm_temp = self._normalize_temperature(temperature)
            power_hist = self.power_history[-self.observation_history_length:]
            temp_hist = self.temperature_history[-self.observation_history_length:]
            
            norm_power_hist = [self._normalize_power(p) for p in power_hist]
            norm_temp_hist = [self._normalize_temperature(t) for t in temp_hist]
            
            while len(norm_power_hist) < self.observation_history_length:
                norm_power_hist.insert(0, 0.0)
            while len(norm_temp_hist) < self.observation_history_length:
                norm_temp_hist.insert(0, 0.0)
            
            obs_list = [norm_temp] + norm_power_hist + norm_temp_hist
        
        return np.array(obs_list, dtype=np.float32)
    
    def _action_to_power(self, action: np.ndarray) -> float:
        """
        Convert normalized action [-1, 1] to power value.
        
        Action interpretation:
        - -1.0 -> power_min
        - 0.0 -> current_power (no change)
        - +1.0 -> power_max
        
        Or alternatively, action is a direct power adjustment:
        - action * power_range -> power adjustment
        """
        # Option 1: Action is direct power adjustment (relative to current)
        # Use smaller adjustment to prevent large swings
        power_range = self.power_max - self.power_min
        # Reduce adjustment scale from 0.5 to 0.2 for more stable control
        power_adjustment = float(action[0]) * power_range * 0.2  # Scale to ±20% of range
        new_power = self.current_power + power_adjustment
        
        # Option 2: Action is absolute power (normalized)
        # normalized_power = (action[0] + 1.0) / 2.0  # Map [-1, 1] to [0, 1]
        # new_power = self.power_min + normalized_power * power_range
        
        # Clamp to valid range
        new_power = np.clip(new_power, self.power_min, self.power_max)
        return float(new_power)
    
    def _compute_reward(self, temperature: float) -> float:
        """
        Compute reward based on temperature error.
        
        Reward is negative squared error from target temperature, normalized.
        """
        error = self.target_temperature - temperature
        # Normalize error by target temperature to keep rewards in reasonable range
        normalized_error = error / self.target_temperature
        # Scale reward to be in range [-1, 0] when error is small
        # Use smaller scale to prevent explosion
        reward = -self.reward_scale * (normalized_error ** 2)
        # Clip reward to prevent extreme values
        reward = np.clip(reward, -100.0, 0.0)
        return float(reward)
    
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict] = None
    ) -> Tuple[np.ndarray, Dict]:
        """
        Reset the environment to initial state.
        
        Returns:
        --------
        observation : np.ndarray
            Initial observation (normalized temperature)
        info : dict
            Additional information
        """
        super().reset(seed=seed)
        
        # Reset environment state
        self.current_power = self.base_power
        self.step_count = 0
        self.episode_reward = 0.0
        self.episode_length = 0
        self.simulation_complete = False
        
        # Reset history
        self.power_history.clear()
        self.temperature_history.clear()
        
        # Reset temperature state
        # Add some randomness to initial temperature for exploration
        initial_temp_ratio = 0.85 + np.random.uniform(0, 0.15)  # 85-100% of target
        self._current_temp = self.target_temperature * initial_temp_ratio
        
        # Reset power controller in config
        self.solver_input["power_controller"]["controller_params"]["base_power"] = self.base_power
        
        # Get initial observation with history (all zeros for history initially)
        initial_temp = self._current_temp
        observation = self._get_observation(initial_temp, self.current_power)
        self._last_obs = observation
        
        info = {
            "episode": {
                "r": 0.0,
                "l": 0
            }
        }
        
        return observation, info
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one step in the environment.
        
        This is a simplified step function that models the temperature response.
        For actual training, you would need to run the simulation incrementally.
        
        Parameters:
        -----------
        action : np.ndarray
            Normalized action in [-1, 1] range
            
        Returns:
        --------
        observation : np.ndarray
            Next observation (normalized temperature)
        reward : float
            Reward for this step
        terminated : bool
            Whether episode terminated (simulation ended)
        truncated : bool
            Whether episode was truncated (max steps reached)
        info : dict
            Additional information
        """
        # Convert action to power
        # Ensure action is in valid range
        action = np.clip(action, -1.0, 1.0)
        new_power = self._action_to_power(action)
        self.current_power = new_power
        
        # Simplified temperature dynamics model
        # This models the relationship between power and temperature
        # In a real implementation, this would come from the actual simulation
        
        # More stable temperature dynamics model
        # Temperature response to power: T ∝ P^α where α ≈ 0.5-0.7 for thermal systems
        power_ratio = self.current_power / self.base_power if self.base_power > 0 else 1.0
        
        # Model temperature response with some dynamics
        if not hasattr(self, '_current_temp'):
            self._current_temp = self.target_temperature * 0.9
        
        # Target temperature based on power (non-linear relationship)
        # Use power law: T_target = T_base * (P/P_base)^alpha
        alpha = 0.6  # Power law exponent (typical for thermal systems)
        target_temp_from_power = self.target_temperature * (power_ratio ** alpha)
        
        # First-order response: temp moves toward target with time constant
        # Use smaller time constant for stability
        time_constant = 0.05  # Reduced from 0.1 for more stable dynamics
        temp_change = (target_temp_from_power - self._current_temp) * time_constant
        self._current_temp += temp_change
        
        # Clamp temperature to reasonable range to prevent instability
        self._current_temp = np.clip(self._current_temp, 500.0, 5000.0)
        
        # Add small noise for realism (reduced from 10K)
        noise = np.random.normal(0, 5.0)  # 5K noise
        current_temp = self._current_temp + noise
        current_temp = np.clip(current_temp, 500.0, 5000.0)
        
        # Update history (before getting observation)
        self.power_history.append(self.current_power)
        self.temperature_history.append(current_temp)
        
        # Get observation with history
        observation = self._get_observation(current_temp, self.current_power)
        self._last_obs = observation
        
        # Compute reward
        reward = self._compute_reward(current_temp)
        self.episode_reward += reward
        self.episode_length += 1
        self.step_count += 1
        
        # Check termination conditions
        # If max_steps is None, use a default long episode length for training
        # This prevents episodes from running indefinitely
        effective_max_steps = self.max_steps if self.max_steps is not None else 10000
        
        if self.step_count >= effective_max_steps:
            terminated = True
            truncated = True
        else:
            terminated = self.simulation_complete
            truncated = False
        
        info = {
            "temperature": float(current_temp),
            "power": float(self.current_power),
            "episode": {
                "r": self.episode_reward,
                "l": self.episode_length
            }
        }
        
        return observation, reward, terminated, truncated, info
    
    def render(self):
        """Render the environment (not implemented)."""
        if self.render_mode == "human":
            print(f"Step: {self.step_count}, Power: {self.current_power:.2f}W, "
                  f"Temp: {self._denormalize_temperature(self._last_obs[0]):.2f}K")
    
    def close(self):
        """Clean up environment resources."""
        pass

