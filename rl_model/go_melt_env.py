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
import tempfile
import random
from typing import Dict, Tuple, Optional, Any, List
from pathlib import Path
import gymnasium as gym
from gymnasium import spaces

# Add parent directory to path for imports
_current_dir = os.path.dirname(os.path.abspath(__file__))
_parent_dir = os.path.dirname(_current_dir)
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

# Import GoMeltSimulator
try:
    from simulator.simulator import GoMeltSimulator
    HAS_SIMULATOR = True
except ImportError as e:
    print(f"Warning: Could not import GoMeltSimulator: {e}")
    HAS_SIMULATOR = False
    GoMeltSimulator = None

try:
    import jax.numpy as jnp
    HAS_JAX = True
except ImportError:
    HAS_JAX = False
    jnp = None

# Import functions from controller (for backward compatibility and direct access)
try:
    from controller import (
        get_max_temperature_level3, 
        get_fraction_above_liquidus,
        get_accum_time_local,
        get_laser_speed,
        get_distance_to_last_point,
        get_temperature_stats_level3
    )
    get_max_temp_L3 = get_max_temperature_level3
    get_frac_above_liquidus = get_fraction_above_liquidus
    get_accum_time_local_func = get_accum_time_local
    get_laser_speed_func = get_laser_speed
    get_distance_to_last_point_func = get_distance_to_last_point
    get_temperature_stats_L3 = get_temperature_stats_level3
except ImportError:
    get_max_temp_L3 = None
    get_frac_above_liquidus = None
    get_accum_time_local_func = None
    get_laser_speed_func = None
    get_distance_to_last_point_func = None
    get_temperature_stats_L3 = None

# Import observation utilities (from same directory)
# Use absolute imports - add current directory to path if needed
_current_dir = os.path.dirname(os.path.abspath(__file__))
if _current_dir not in sys.path:
    sys.path.insert(0, _current_dir)

from rl_model.observation_utils import (
    build_observation,
    calculate_observation_size,
    print_observation_breakdown
)
HAS_OBSERVATION_UTILS = True


# Class-level flag to track if we've already warned about missing simulator
_has_warned_simulator = False


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
        observation_history_length: int = 10,
        verbose: bool = False
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
        verbose : bool
            Whether to print step details from simulator. Default: False
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
        self.verbose = verbose
        
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
        
        # History tracking for observations
        self.power_history = []
        self.temperature_history = []
        
        # Initialize persistent simulator (like Isaac Lab)
        # The simulator will persist between resets
        # Use lazy initialization to avoid blocking during environment creation
        self.simulator = None
        self._simulator_initialized = False
        
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
    def _get_current_laser_position(self) -> Optional[np.ndarray]:
        """
        Get current laser position from simulator state.
        Returns None if simulator is not available.
        """
        if self.simulator is None or not hasattr(self.simulator, 'Levels'):
            return None
        
        # Try to get position from toolpath file (simulator reads it)
        # For now, return None and let simulator handle position internally
        return None
    
    def _get_laser_all_from_simulator(self) -> Optional[np.ndarray]:
        """
        Construct laser_all array from simulator state for observation functions.
        Returns None if simulator is not available.
        """
        if self.simulator is None:
            return None
        
        # The simulator reads toolpath internally, so we can't easily get the last positions
        # Return None and let observation functions handle it
        return None
    
    def _calculate_observation_size(self) -> int:
        """
        Calculate the observation size based on the observation_config.
        Uses observation_utils if available, otherwise falls back to manual calculation.
        """
        if HAS_OBSERVATION_UTILS and calculate_observation_size is not None:
            return calculate_observation_size(
                self.observation_config,
                self.observation_history_length
            )
        else:
            # Fallback calculation
            size = 0
            if self.observation_config:
                if self.observation_config.get('enable_max_T_L3', True):
                    size += 1
                if self.observation_config.get('enable_frac_above_liquidus', False):
                    size += 1
                if self.observation_config.get('enable_power_history', True):
                    size += self.observation_config.get('power_history_length', self.observation_history_length)
                if self.observation_config.get('enable_temperature_history', True):
                    size += self.observation_config.get('temperature_history_length', self.observation_history_length)
            if size == 0:
                size = 1 + 2 * self.observation_history_length
            return size
    
    def _initialize_simulator(self):
        """Initialize the persistent GoMeltSimulator instance."""
        global _has_warned_simulator
        if not HAS_SIMULATOR or GoMeltSimulator is None:
            if not _has_warned_simulator:
                print("Warning: GoMeltSimulator not available. Environment will not work properly.")
                _has_warned_simulator = True
            return False
        
        try:
            # Create simulator instance (persistent between resets)
            self.simulator = GoMeltSimulator(self.solver_input, self.config_file, verbose=self.verbose)
            if self.verbose:
                print("✓ GoMeltSimulator initialized successfully")
            return True
        except Exception as e:
            print(f"Warning: Failed to initialize GoMeltSimulator: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _get_observation(self, temperature: float, power: float, laser_all: Optional[np.ndarray] = None, accum_time: Optional[np.ndarray] = None, accum_idx: Optional[int] = None) -> np.ndarray:
        """
        Get observation array with configurable features.
        
        Uses observation_utils.build_observation if available, otherwise falls back
        to simple observation format.
        
        Parameters:
        -----------
        temperature : float
            Current temperature
        power : float
            Current power
        laser_all : np.ndarray, optional
            Array of laser positions for speed/distance calculations
        accum_time : np.ndarray, optional
            Accumulated time array
        accum_idx : int, optional
            Index for accumulated time
            
        Returns:
        --------
        np.ndarray
            Observation array with features specified in observation_config
        """
        if HAS_OBSERVATION_UTILS and build_observation is not None:
            # Get levels and properties from simulator
            levels = None
            if self.simulator is not None and hasattr(self.simulator, 'Levels'):
                levels = self.simulator.Levels
            
            # Get future/history toolpath (currently not available from simulator)
            # TODO: If needed, could read toolpath file separately
            future_toolpath = None
            history_toolpath = None
            
            return build_observation(
                observation_config=self.observation_config,
                temperature=temperature,
                power=power,
                power_history=self.power_history,
                temperature_history=self.temperature_history,
                levels=levels,
                properties=self.properties,
                laser_all=laser_all,
                accum_time=accum_time,
                accum_idx=accum_idx,
                future_toolpath=future_toolpath,
                history_toolpath=history_toolpath,
                power_min=self.power_min,
                power_max=self.power_max,
                temp_min=self.temp_min,
                temp_max=self.temp_max,
                default_history_length=self.observation_history_length,
                toolpath_normalization_range=self.observation_config.get('toolpath_normalization_range', 10.0) if self.observation_config else 10.0
            )
        else:
            # Fallback to simple observation format
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
        
        # Lazy initialization: Initialize simulator if not already done
        if not self._simulator_initialized:
            self._initialize_simulator()
            self._simulator_initialized = True
        
        # Reset simulator (persistent instance, just reset its state)
        if self.simulator is not None:
            try:
                self.simulator.reset(self.solver_input, self.config_file)
            except Exception as e:
                print(f"Warning: Failed to reset simulator: {e}")
                # Reinitialize if reset fails
                self._initialize_simulator()
        
        # Get initial temperature from simulator
        if self.simulator is not None and hasattr(self.simulator, 'Levels'):
            # Get initial temperature from simulator state
            if get_max_temp_L3 is not None:
                initial_temp = float(get_max_temp_L3(self.simulator.Levels))
            else:
                # Fallback: get max from T0 array
                if HAS_JAX:
                    initial_temp = float(jnp.max(self.simulator.Levels[3]["T0"]))
                else:
                    initial_temp = float(np.max(self.simulator.Levels[3]["T0"]))
            self._current_temp = initial_temp
        else:
            # Fallback to mock if simulator not available
            initial_temp_ratio = 0.85 + np.random.uniform(0, 0.15)
            self._current_temp = self.target_temperature * initial_temp_ratio
            initial_temp = self._current_temp
        
        # Reset power controller in config
        self.solver_input["power_controller"]["controller_params"]["base_power"] = self.base_power
        
        # Get initial observation with history (all zeros for history initially)
        laser_all = self._get_laser_all_from_simulator()
        
        accum_time = None
        accum_idx = None
        if self.simulator is not None and hasattr(self.simulator, 'Levels'):
            if isinstance(self.simulator.Levels, list) and len(self.simulator.Levels) > 0:
                accum_idx = self.simulator.Levels[0].get("idx", None)
                accum_time = getattr(self.simulator, 'accum_time', None)
        
        observation = self._get_observation(initial_temp, self.current_power, laser_all, accum_time, accum_idx)
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
        # Lazy initialization: Initialize simulator if not already done
        if not self._simulator_initialized:
            self._initialize_simulator()
            self._simulator_initialized = True
        
        # Convert action to power
        # Ensure action is in valid range
        action = np.clip(action, -1.0, 1.0)
        new_power = self._action_to_power(action)
        self.current_power = new_power
        
        # Run simulation step with actual power value
        # Note: At the end of Gcode, toolpath may run out but power could still be set.
        # The simulator's nextstep() will return False when toolpath is exhausted.
        if self.simulator is not None:
            # Run one step of simulation with actual power (even if 0 or low)
            ongoing = self.simulator.nextstep(self.current_power)
            
            if not ongoing:
                # Simulation completed (gcode steps ran out)
                self.simulation_complete = True
            
            # Get current temperature from simulator
            if hasattr(self.simulator, 'Levels') and self.simulator.Levels is not None:
                if get_max_temp_L3 is not None:
                    current_temp = float(get_max_temp_L3(self.simulator.Levels))
                else:
                    # Fallback: get max from T0 array
                    if HAS_JAX:
                        current_temp = float(jnp.max(self.simulator.Levels[3]["T0"]))
                    else:
                        current_temp = float(np.max(self.simulator.Levels[3]["T0"]))
                self._current_temp = current_temp
            else:
                # Fallback if simulator state not available
                current_temp = self._current_temp if hasattr(self, '_current_temp') else self.target_temperature * 0.9
        else:
            # Fallback to mock if simulator not available
            power_ratio = self.current_power / self.base_power if self.base_power > 0 else 1.0
            
            if not hasattr(self, '_current_temp'):
                self._current_temp = self.target_temperature * 0.9
            
            alpha = 0.6
            target_temp_from_power = self.target_temperature * (power_ratio ** alpha)
            time_constant = 0.05
            temp_change = (target_temp_from_power - self._current_temp) * time_constant
            self._current_temp += temp_change
            self._current_temp = np.clip(self._current_temp, 500.0, 5000.0)
            
            noise = np.random.normal(0, 5.0)
            current_temp = self._current_temp + noise
            current_temp = np.clip(current_temp, 500.0, 5000.0)
            self._current_temp = current_temp
        
        # Construct laser_all array for observation functions
        laser_all = self._get_laser_all_from_simulator()
        
        # Get accum_time and accum_idx if available
        accum_time = None
        accum_idx = None
        if self.simulator is not None and hasattr(self.simulator, 'Levels'):
            if isinstance(self.simulator.Levels, list) and len(self.simulator.Levels) > 0:
                accum_idx = self.simulator.Levels[0].get("idx", None)
                accum_time = getattr(self.simulator, 'accum_time', None)
        
        # Update history (before getting observation)
        self.power_history.append(self.current_power)
        self.temperature_history.append(current_temp)
        
        # Get observation with history
        observation = self._get_observation(current_temp, self.current_power, laser_all, accum_time, accum_idx)
        self._last_obs = observation
        
        # Compute reward
        reward = self._compute_reward(current_temp)
        self.episode_reward += reward
        self.episode_length += 1
        self.step_count += 1
        
        # Check termination conditions
        # Terminate when simulation completes (gcode steps run out)
        if self.simulation_complete:
            terminated = True
            truncated = False
        elif self.max_steps is not None and self.step_count >= self.max_steps:
            # Also check max_steps if specified
            terminated = False
            truncated = True
        else:
            # Continue simulation
            terminated = False
            truncated = False
        
        # Get simulator statistics for info
        sim_stats = {}
        if self.simulator is not None:
            try:
                sim_stats = self.simulator.get_statistics()
            except:
                pass
        
        info = {
            "temperature": float(current_temp),
            "power": float(self.current_power),
            "simulation_complete": self.simulation_complete,
            "time_inc": sim_stats.get("time_inc", self.step_count),
            "total_t_inc": sim_stats.get("total_t_inc", 0),
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
        # Finalize simulator if it exists
        if self.simulator is not None:
            try:
                self.simulator.finalize()
            except:
                pass
            self.simulator = None

