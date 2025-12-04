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
if _current_dir not in sys.path:
    sys.path.insert(0, _current_dir)

try:
    from createPath import parsingGcode
except ImportError:
    parsingGcode = None

# Import GO-MELT simulation functions
try:
    from computeFunctions import (
        SetupProperties, SetupLevels, SetupNonmesh,
        getStaticNodesAndElements, getStaticSubcycle,
        interpolatePointsMatrix, calcStaticTmpNodesAndElements,
        stepGOMELT, moveEverything,
        get_max_temperature_level3 as get_max_temp_L3,
        getSubstrateNodes
    )
    import jax.numpy as jnp
    HAS_GO_MELT = True
except ImportError as e:
    print(f"Warning: Could not import GO-MELT simulation functions: {e}")
    HAS_GO_MELT = False
    SetupProperties = None
    SetupLevels = None
    SetupNonmesh = None

# Import reusable functions from go_melt.py
try:
    from go_melt import (
        initialize_simulation_setup,
        initialize_interpolation,
        handle_layer_change
    )
    HAS_GO_MELT_HELPERS = True
except ImportError as e:
    print(f"Warning: Could not import GO-MELT helper functions from go_melt.py: {e}")
    print("  Falling back to manual initialization.")
    HAS_GO_MELT_HELPERS = False
    initialize_simulation_setup = None
    initialize_interpolation = None
    handle_layer_change = None


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
        
        # GO-MELT simulation state (initialized in reset)
        self.sim_properties = None
        self.sim_nonmesh = None
        self.sim_ne_nn = None
        self.sim_subcycle = None
        self.sim_shapes = None
        self.sim_linterp = None
        self.sim_tmp_ne_nn = None
        self.sim_substrate = None
        self.sim_laser_prev_z = None
        self.sim_move_hist = None
        self.sim_time_inc = 0
        self.sim_toolpath_file = None
        self.sim_toolpath_file_path = None
        self.toolpath_file_path = None  # Path to the generated toolpath file (keep for simulation)
        # Mesh ratios for movement logic (stored to avoid recalculation)
        self.sim_L1L2Eratio = None
        self.sim_L2L3Eratio = None
        
        # History tracking for observations
        self.power_history = []
        self.temperature_history = []
        
        # Load toolpath from gcode file using createPath logic
        # This creates a consistent path list that will be followed in order
        self.toolpath_list = self._generate_toolpath_list()
        self.toolpath_position = 0  # Current position index in toolpath_list (0 to len-1)
        self.toolpath_completed = False  # Track if we've completed the full path
        
        # Load toolpath_step_size and N23 from config
        if "nonmesh" in self.solver_input:
            nonmesh = self.solver_input["nonmesh"]
            laser_velocity = nonmesh.get("laser_velocity", 1000.0)  # mm/s
            timestep_L3 = nonmesh.get("timestep_L3", 1e-5)  # seconds
            self.toolpath_step_size = laser_velocity * timestep_L3  # mm per step
            
            # Calculate N23 = subcycle_num_L2 * subcycle_num_L3
            subcycle_num_L2 = nonmesh.get("subcycle_num_L2", 1)
            subcycle_num_L3 = nonmesh.get("subcycle_num_L3", 1)
            self.N23 = subcycle_num_L2 * subcycle_num_L3
        else:
            self.toolpath_step_size = 0.1  # Default: 0.1 mm per step
            self.N23 = 1  # Default: 1
        
        print(f"Initialized toolpath with {len(self.toolpath_list)} points")
        print(f"  N23 (step spacing): {self.N23}")
        
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
    #checked
    def _generate_toolpath_list(self) -> List[np.ndarray]:
        """
        Generate full toolpath list from gcode file using createPath.parsingGcode.
        This creates an interpolated toolpath with points spaced by toolpath_step_size.
        
        Returns:
        --------
        List[np.ndarray]
            List of toolpath points as numpy arrays [x, y, z]
        """
        # Get config parameters
        nonmesh = self.solver_input.get("nonmesh", {})
        properties = self.solver_input.get("properties", {})
        gcode_path = nonmesh.get("gcode")
        
        # Create temporary gcode file if needed
        tmp_gcode_path = None
        if not gcode_path or not os.path.exists(gcode_path):
            # Create a temporary gcode file with default pattern
            random_X_Start = np.random.uniform(0.0, 10.0)
            random_X_End = np.random.uniform(0.0, 10.0)
            Y = 0.0
            Z= 0.04
            repeat_times = random.randint(2, 10)

            with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.gcode') as tmp_gcode_file:
                tmp_gcode_path = tmp_gcode_file.name
                # Write default pattern: (2.0, 2.0, 0.04) -> (8.0, 2.0, 0.04) -> repeat
                tmp_gcode_file.write(f"G0 X{random_X_Start} Y{Y} Z{Z}\n")
                for i in range(repeat_times):
                    tmp_gcode_file.write(f"G1 X{random_X_End} Y{Y} Z{Z}\n")
                    tmp_gcode_file.write(f"G1 X{random_X_Start} Y{Y} Z{Z}\n")   
            print(f"Created temporary gcode file with default pattern: {tmp_gcode_path}")
            gcode_path = tmp_gcode_path
        
        # Use createPath.parsingGcode to generate toolpath
        try:
            # Create a temporary toolpath file
            with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as tmp_file:
                tmp_toolpath = tmp_file.name
            
            # Create a temporary nonmesh config for parsingGcode
            tmp_nonmesh = nonmesh.copy()
            tmp_nonmesh["toolpath"] = tmp_toolpath
            tmp_nonmesh["gcode"] = gcode_path  # Use the gcode path (original or temp)
            
            # L2h is not used in parsingGcode, so we can pass a dummy value
            # (it's only used in the actual simulation, not in path generation)
            dummy_L2h = 0.0
            
            # Generate toolpath using createPath
            parsingGcode(tmp_nonmesh, properties, dummy_L2h)
            
            # Read the generated toolpath file
            toolpath_list = []
            with open(tmp_toolpath, 'r') as f:
                for line in f:
                    parts = line.strip().split(',')
                    if len(parts) >= 3:
                        x = float(parts[0])
                        y = float(parts[1])
                        z = float(parts[2])
                        toolpath_list.append(np.array([x, y, z], dtype=np.float32))
            
            # Keep the toolpath file for simulation (don't delete it)
            self.toolpath_file_path = tmp_toolpath
            
            # Clean up temporary gcode file if we created it
            if tmp_gcode_path:
                try:
                    os.unlink(tmp_gcode_path)
                except:
                    pass
            
            if len(toolpath_list) > 0:
                print(f"Generated {len(toolpath_list)} toolpath points using createPath.parsingGcode")
                print(f"Toolpath file saved at: {self.toolpath_file_path}")
                return toolpath_list
            else:
                raise ValueError("parsingGcode generated empty toolpath")
                
        except Exception as e:
            print(f"Warning: Failed to use createPath.parsingGcode: {e}")
            import traceback
            traceback.print_exc()
            
            # Clean up temporary gcode file if created
            if tmp_gcode_path:
                try:
                    os.unlink(tmp_gcode_path)
                except:
                    pass
            
            # Fallback: return empty list (should not happen, but handle gracefully)
            print("Error: Could not generate toolpath. Returning empty list.")
            exit()
    #checked
    def _get_current_toolpath_position(self) -> np.ndarray:
        """
        Get current toolpath position using index.
        Returns the actual position from the path list (no wrapping).
        
        Returns:
        --------
        np.ndarray
            Current position [x, y, z] in mm
        """
        if len(self.toolpath_list) == 0:
            return np.array([0.0, 0.0, 0.0], dtype=np.float32)
        
        # Clamp to valid range (don't wrap)
        idx = min(self.toolpath_position, len(self.toolpath_list) - 1)
        return self.toolpath_list[idx].copy()
    #checked
    def _get_future_toolpath(self, length: int) -> np.ndarray:
        """
        Get future toolpath points relative to current position using indices.
        
        Parameters:
        -----------
        length : int
            Number of points to return
            
        Returns:
        --------
        np.ndarray
            Array of shape (length, 3) with relative positions [x, y, z] in mm
        """
        if len(self.toolpath_list) == 0:
            return np.zeros((length, 3), dtype=np.float32)
        
        current_pos = self._get_current_toolpath_position()
        future_points = []
        
        # Space points by N23 from config
        step_spacing = self.N23
        
        for i in range(length):
            # Calculate future index (don't wrap, clamp to end of path)
            future_idx = min(self.toolpath_position + (i + 1) * step_spacing, len(self.toolpath_list) - 1)
            future_pos = self.toolpath_list[future_idx]
            # Return relative to current position
            relative_pos = future_pos - current_pos
            future_points.append(relative_pos)
        
        return np.array(future_points, dtype=np.float32)   
    #checked
    def _get_history_toolpath(self, length: int) -> np.ndarray:
        """
        Get history toolpath points relative to current position using indices.
        
        Parameters:
        -----------
        length : int
            Number of points to return
            
        Returns:
        --------
        np.ndarray
            Array of shape (length, 3) with relative positions [x, y, z] in mm
        """
        if len(self.toolpath_list) == 0:
            return np.zeros((length, 3), dtype=np.float32)
        
        current_pos = self._get_current_toolpath_position()
        history_points = []
        
        # Space points by N23 from config
        step_spacing = self.N23
        
        for i in range(length):
            # Calculate past index (clamp to 0, don't wrap)
            past_idx = max(0, self.toolpath_position - (i + 1) * step_spacing)
            past_pos = self.toolpath_list[past_idx]
            
            # Return relative to current position
            relative_pos = past_pos - current_pos
            history_points.append(relative_pos)
        
        # Reverse so oldest is first
        history_points.reverse()
        return np.array(history_points, dtype=np.float32)
    
    def _initialize_simulation(self):
        """
        Initialize the real GO-MELT simulation state.
        This sets up Levels, Properties, Nonmesh, and all necessary structures.
        Uses reusable functions from go_melt.py when available.
        """
        if not HAS_GO_MELT:
            print("Warning: GO-MELT simulation functions not available. Using mock simulation.")
            return False
        
        try:
            # Use reusable initialization function if available, otherwise fallback
            if HAS_GO_MELT_HELPERS and initialize_simulation_setup is not None:
                # Use refactored initialization function
                Properties, Levels, Nonmesh, ne_nn, subcycle, L1L2Eratio, L2L3Eratio = initialize_simulation_setup(self.solver_input)
                
                self.sim_properties = Properties
                self.sim_nonmesh = Nonmesh
                self.levels = Levels
                self.sim_ne_nn = ne_nn
                self.sim_subcycle = subcycle
                self.sim_L1L2Eratio = L1L2Eratio
                self.sim_L2L3Eratio = L2L3Eratio
                
                # Use reusable interpolation function
                self.sim_linterp = initialize_interpolation(self.levels) if initialize_interpolation is not None else None
            else:
                # Fallback to manual initialization if helpers not available
                self.sim_properties = SetupProperties(self.solver_input.get("properties", {}))
                self.sim_nonmesh = SetupNonmesh(self.solver_input.get("nonmesh", {}))
                self.levels = SetupLevels(self.solver_input, self.sim_properties)
                self.sim_ne_nn = getStaticNodesAndElements(self.levels)
                self.sim_subcycle = getStaticSubcycle(self.sim_nonmesh)
                
                # Calculate mesh ratios manually
                self.sim_L1L2Eratio = [
                    int(jnp.round(self.levels[1]["h"][i] / self.levels[2]["h"][i])) for i in range(2)
                ] + [int(jnp.round(self.sim_properties["layer_height"] / self.levels[2]["h"][2]))]
                self.sim_L2L3Eratio = [
                    int(jnp.round(self.levels[2]["h"][i] / self.levels[3]["h"][i])) for i in range(3)
                ]
            
            # Initialize interpolation if not already done
            if self.sim_linterp is None:
                L1L2Interp = interpolatePointsMatrix(self.levels[1], self.levels[2]["node_coords"])
                L2L3Interp = interpolatePointsMatrix(self.levels[2], self.levels[3]["node_coords"])
                self.sim_linterp = [L1L2Interp, L2L3Interp]
            
            # Initialize Shapes (will be set by moveEverything)
            self.sim_shapes = None
            
            # Initialize substrate using getSubstrateNodes
            self.sim_substrate = getSubstrateNodes(self.levels)
            
            # Initialize tracking variables
            self.sim_laser_prev_z = float("inf")
            self.sim_move_hist = [jnp.array(0), jnp.array(0), jnp.array(0)]
            self.sim_time_inc = 0
            
            # Use the toolpath file generated by _generate_toolpath_list
            # Update nonmesh to point to the generated toolpath file
            if self.toolpath_file_path and os.path.exists(self.toolpath_file_path):
                self.sim_nonmesh["toolpath"] = self.toolpath_file_path
                self.sim_toolpath_file_path = self.toolpath_file_path
                self.sim_toolpath_file = open(self.sim_toolpath_file_path, "r")
            elif self.sim_nonmesh.get("toolpath"):
                # Fallback to original toolpath path
                self.sim_toolpath_file_path = self.sim_nonmesh["toolpath"]
                if os.path.exists(self.sim_toolpath_file_path):
                    self.sim_toolpath_file = open(self.sim_toolpath_file_path, "r")
                else:
                    print(f"Warning: Toolpath file not found: {self.sim_toolpath_file_path}")
                    return False
            else:
                print("Warning: No toolpath file available for simulation")
                return False
            
            print("✓ GO-MELT simulation initialized successfully")
            return True
            
        except Exception as e:
            print(f"Warning: Failed to initialize GO-MELT simulation: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _run_simulation_step(self, laser_position: np.ndarray, laser_power: float) -> Optional[float]:
        """
        Run a single step of the GO-MELT simulation.
        
        Parameters:
        -----------
        laser_position : np.ndarray
            Current laser position [x, y, z, jump, dwell, dt, power]
        laser_power : float
            Current laser power in Watts
            
        Returns:
        --------
        float or None
            Maximum temperature in Level 3, or None if simulation failed
        """
        if not HAS_GO_MELT or self.levels is None:
            return None
        
        try:
            # Convert laser position to JAX array format
            # Format: [x, y, z, jump, dwell, dt, power]
            dt = self.sim_nonmesh.get("timestep_L3", 1e-5)
            laser_pos_jax = jnp.array([
                float(laser_position[0]),  # x
                float(laser_position[1]),  # y
                float(laser_position[2]),  # z
                1.0,  # jump (1 = normal move, 0 = jump)
                1.0,  # dwell (1 = normal, 0 = dwell)
                dt,   # timestep
                laser_power  # power
            ], dtype=jnp.float32)
            
            # Track if layer change was handled
            layer_changed = False
            
            # Handle layer changes if needed
            if laser_pos_jax[2] != self.sim_laser_prev_z:
                layer_changed = True
                # Use reusable layer change handler if available
                if HAS_GO_MELT_HELPERS and handle_layer_change is not None:
                    # Use handle_layer_change for proper layer handling
                    # Note: We pass None for accum_time/max_accum_time since environment doesn't track these
                    self.sim_linterp, self.sim_tmp_ne_nn, self.sim_laser_prev_z, _, _, _, _, _ = handle_layer_change(
                        laser_pos_jax, self.sim_laser_prev_z, self.levels, self.sim_properties,
                        self.sim_linterp, self.sim_nonmesh, False, False, None, None
                    )
                else:
                    # Fallback: just update the previous z
                    self.sim_laser_prev_z = laser_pos_jax[2]
            
            # Move meshes if needed (first time or layer change)
            if self.sim_shapes is None or layer_changed:
                # Get initial laser position
                laser_start = jnp.array([
                    float(laser_position[0]),
                    float(laser_position[1]),
                    float(laser_position[2]),
                    0.0, 0.0, 0.0, 0.0
                ], dtype=jnp.float32)
                
                # Move meshes using stored ratios (always set during initialization)
                (self.levels, self.sim_shapes, self.sim_linterp, self.sim_move_hist) = moveEverything(
                    laser_pos_jax,
                    laser_start,
                    self.levels,
                    self.sim_move_hist,
                    self.sim_linterp,
                    self.sim_L1L2Eratio,
                    self.sim_L2L3Eratio,
                    self.sim_properties["layer_height"],
                )
                
                # Update tmp_ne_nn if not already updated by handle_layer_change
                if not (layer_changed and HAS_GO_MELT_HELPERS and handle_layer_change is not None):
                    self.sim_tmp_ne_nn = calcStaticTmpNodesAndElements(self.levels, laser_pos_jax)
            
            # Run simulation step
            # Always use stepGOMELT for single step (subcycle would need multiple positions)
            self.levels, _ = stepGOMELT(
                self.levels,
                self.sim_ne_nn,
                self.sim_tmp_ne_nn,
                self.sim_shapes,
                self.sim_linterp,
                laser_pos_jax[:3],  # position only [x, y, z]
                self.sim_properties,
                dt,
                laser_power,
                self.sim_substrate
            )
            
            # Get maximum temperature from Level 3
            if get_max_temp_L3 is not None:
                max_temp = float(get_max_temp_L3(self.levels))
            else:
                # Fallback: get max from T0 array
                max_temp = float(jnp.max(self.levels[3]["T0"]))
            
            self.sim_time_inc += 1
            return max_temp
            
        except Exception as e:
            print(f"Warning: Simulation step failed: {e}")
            import traceback
            traceback.print_exc()
            return None
    
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
            
            # Future toolpath (relative positions, normalized within config range)
            if self.observation_config.get('enable_future_toolpath', False):
                future_toolpath_len = self.observation_config.get('future_toolpath_length', 10)
                future_path = self._get_future_toolpath(future_toolpath_len)  # Already relative to current position
                # Normalize relative path within config-defined range
                toolpath_range = self.observation_config.get('toolpath_normalization_range', 10.0)  # Default: ±10mm
                future_path_flat = future_path.flatten()
                # Normalize: divide by range, clip to [-1, 1], then convert to [0, 1]
                normalized = np.clip(future_path_flat / toolpath_range, -1.0, 1.0)
                normalized = (normalized + 1.0) / 2.0  # Convert to [0, 1]
                obs_list.extend(normalized.tolist())
            
            # History toolpath (relative positions, normalized within config range)
            if self.observation_config.get('enable_history_toolpath', False):
                history_toolpath_len = self.observation_config.get('history_toolpath_length', 10)
                hist_path = self._get_history_toolpath(history_toolpath_len)  # Already relative to current position
                # Normalize relative path within config-defined range
                toolpath_range = self.observation_config.get('toolpath_normalization_range', 10.0)  # Default: ±10mm
                hist_path_flat = hist_path.flatten()
                # Normalize: divide by range, clip to [-1, 1], then convert to [0, 1]
                normalized = np.clip(hist_path_flat / toolpath_range, -1.0, 1.0)
                normalized = (normalized + 1.0) / 2.0  # Convert to [0, 1]
                obs_list.extend(normalized.tolist())
        
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
        
        # Reset toolpath position and completion flag
        self.toolpath_position = 0
        self.toolpath_completed = False
        
        # Close existing toolpath file if open
        if self.sim_toolpath_file is not None:
            try:
                self.sim_toolpath_file.close()
            except:
                pass
            self.sim_toolpath_file = None
        
        # Initialize or reinitialize GO-MELT simulation
        if not self._initialize_simulation():
            # Fallback to mock if simulation initialization fails
            initial_temp_ratio = 0.85 + np.random.uniform(0, 0.15)
            self._current_temp = self.target_temperature * initial_temp_ratio
            initial_temp = self._current_temp
        else:
            # Get initial temperature from simulation
            if len(self.toolpath_list) > 0:
                initial_laser_pos = self.toolpath_list[0]
                initial_temp = self._run_simulation_step(initial_laser_pos, self.current_power)
                if initial_temp is None:
                    # Fallback if simulation step fails
                    initial_temp_ratio = 0.85 + np.random.uniform(0, 0.15)
                    self._current_temp = self.target_temperature * initial_temp_ratio
                    initial_temp = self._current_temp
                else:
                    self._current_temp = initial_temp
            else:
                initial_temp = self.target_temperature * 0.9
                self._current_temp = initial_temp
        
        # Reset power controller in config
        self.solver_input["power_controller"]["controller_params"]["base_power"] = self.base_power
        
        # Get initial observation with history (all zeros for history initially)
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
        
        # Get current toolpath position before advancing
        if len(self.toolpath_list) > 0 and self.toolpath_position < len(self.toolpath_list):
            current_laser_pos = self.toolpath_list[self.toolpath_position]
        else:
            # Fallback if no toolpath
            current_laser_pos = np.array([0.0, 0.0, 0.04], dtype=np.float32)
        
        # Run real GO-MELT simulation step
        current_temp = self._run_simulation_step(current_laser_pos, self.current_power)
        
        # Fallback to mock if simulation fails
        if current_temp is None:
            # Simplified temperature dynamics model (fallback)
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
        else:
            # Update tracked temperature from real simulation
            self._current_temp = current_temp
        
        # Advance toolpath position (follow path in order, don't wrap)
        if len(self.toolpath_list) > 0:
            self.toolpath_position += 1
            # Check if we've completed the full path
            if self.toolpath_position >= len(self.toolpath_list):
                self.toolpath_completed = True
                self.toolpath_position = len(self.toolpath_list) - 1  # Keep at last position
        else:
            self.toolpath_position += 1
        
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
        # Terminate when we've completed the full toolpath
        if self.toolpath_completed:
            terminated = True
            truncated = False
        elif self.max_steps is not None and self.step_count >= self.max_steps:
            # Also check max_steps if specified
            terminated = False
            truncated = True
        else:
            # Continue following the path
            terminated = self.simulation_complete
            truncated = False
        
        # Get current toolpath position for info
        current_toolpath_pos = None
        if len(self.toolpath_list) > 0 and self.toolpath_position < len(self.toolpath_list):
            current_toolpath_pos = self.toolpath_list[self.toolpath_position]
        
        info = {
            "temperature": float(current_temp),
            "power": float(self.current_power),
            "toolpath_position": int(self.toolpath_position),
            "toolpath_progress": float(self.toolpath_position / len(self.toolpath_list)) if len(self.toolpath_list) > 0 else 0.0,
            "toolpath_completed": self.toolpath_completed,
            "current_toolpath_point": current_toolpath_pos.tolist() if current_toolpath_pos is not None else None,
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
        # Close toolpath file if open
        if self.sim_toolpath_file is not None:
            try:
                self.sim_toolpath_file.close()
            except:
                pass
            self.sim_toolpath_file = None
        
        # Clean up temporary toolpath file if we created it
        if self.toolpath_file_path and os.path.exists(self.toolpath_file_path):
            # Check if it's a temporary file (in temp directory)
            if tempfile.gettempdir() in self.toolpath_file_path:
                try:
                    os.unlink(self.toolpath_file_path)
                except:
                    pass

