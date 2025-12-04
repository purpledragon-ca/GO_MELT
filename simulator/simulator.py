import os
import sys
import time
import copy
import shutil
from pathlib import Path

import dill
import jax
import jax.numpy as jnp
import numpy as np
from computeFunctions import *
from createPath import parsingGcode, count_lines
import gc
import json
from datetime import datetime


class GoMeltSimulator:
    """
    GO-MELT Simulation Environment
    
    Each instance represents a complete simulation environment for a single gcode file.
    The simulator manages all state, configuration, and execution of the thermal simulation.
    For each step, it will require a power input.
    """
    
    def __init__(self, solver_input: dict, input_file: str = None):
        """
        Initialize the simulator with configuration.
        
        Parameters:
        -----------
        solver_input : dict
            Complete simulation configuration dictionary
        input_file : str | None
            Path to input JSON file (optional, for output naming)
        """
        # Store configuration
        self.solver_input = solver_input
        self.input_file = input_file
        
        # Initialize all components
        self._initialize_simulation_setup()
        self._setup_save_path()
        self._initialize_toolpath()
        self._initialize_interpolation()
        self._initialize_time_tracking()
        
        # Open toolpath file
        self.tool_path_file = open(self.Nonmesh["toolpath"], "r")
        self.np_path = Path(self.Nonmesh["save_path"] + "checkpoint").absolute()
        self.layer_check = self.Nonmesh["layer_num"] + self.Nonmesh["restart_layer_num"]
        
        # Flag for first moveEverything call
        self._first_move = False
        
        # Flag to track if initialized
        self._initialized = False
        
        # Buffer for reading toolpath lines
        self._toolpath_buffer = []
        self._buffer_index = 0
        
        # Save initial results
        saveResults(self.Levels, self.Nonmesh, self.savenum, power=self.initial_power)
    
    def _initialize_simulation_setup(self):
        """Initialize Properties, Levels, Nonmesh, and compute mesh ratios."""
        Properties = SetupProperties(self.solver_input.get("properties", {}))
        Levels = SetupLevels(self.solver_input, Properties)
        Nonmesh = SetupNonmesh(self.solver_input.get("nonmesh", {}))
        
        # Static Mesh Metadata
        ne_nn = getStaticNodesAndElements(Levels)
        subcycle = getStaticSubcycle(Nonmesh)
        
        # Mesh Ratios for Movement Logic
        L1L2Eratio = [
            int(jnp.round(Levels[1]["h"][i] / Levels[2]["h"][i])) for i in range(2)
        ] + [int(jnp.round(Properties["layer_height"] / Levels[2]["h"][2]))]
        
        L2L3Eratio = [
            int(jnp.round(Levels[2]["h"][i] / Levels[3]["h"][i])) for i in range(3)
        ]
        
        # Store as attributes
        self.Properties = Properties
        self.Levels = Levels
        self.Nonmesh = Nonmesh
        self.ne_nn = ne_nn
        self.subcycle = subcycle
        self.L1L2Eratio = L1L2Eratio
        self.L2L3Eratio = L2L3Eratio
    
    def _setup_save_path(self):
        """Setup save path with timestamp and update Nonmesh."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if self.input_file:
            json_name = Path(self.input_file).stem
        else:
            json_name = "simulation"
        
        base_save_path = self.Nonmesh.get("save_path", "./results/")
        base_path = Path(base_save_path.rstrip("/")).parent
        new_save_path = base_path / f"{timestamp}_{json_name}"
        
        new_save_path.mkdir(parents=True, exist_ok=True)
        new_save_path_str = str(new_save_path) + "/"
        
        self.Nonmesh["save_path"] = new_save_path_str
        
        original_toolpath = self.Nonmesh.get("toolpath", "./results/example/toolpath.txt")
        toolpath_filename = Path(original_toolpath).name
        new_toolpath = new_save_path / toolpath_filename
        
        if self.Nonmesh.get("use_txt", 0):
            original_toolpath_path = Path(original_toolpath)
            if original_toolpath_path.exists():
                shutil.copy2(original_toolpath_path, new_toolpath)
                print(f"Copied toolpath file to: {new_toolpath}")
            else:
                print(f"Warning: Toolpath file not found at {original_toolpath}")
        
        self.Nonmesh["toolpath"] = str(new_toolpath)
        
        print(f"Results will be saved to: {new_save_path_str}")
    
    def _initialize_toolpath(self):
        """Parse toolpath and get initial laser position."""
        if self.Nonmesh["use_txt"]:
            move_mesh = count_lines(self.Nonmesh["toolpath"])
        else:
            move_mesh = parsingGcode(self.Nonmesh, self.Properties, self.Levels[2]["h"])
        
        self.total_t_inc = move_mesh
        
        if not self.Properties["laser_center"]:
            with open(self.Nonmesh["toolpath"], "r") as tool_path_file:
                self.laser_start = np.array(
                    [float(val) for val in tool_path_file.readline().split(",")]
                )
        else:
            self.laser_start = np.array(self.Properties["laser_center"])
    
    def _initialize_interpolation(self):
        """Initialize interpolation matrices between levels."""
        L1L2Interp = interpolatePointsMatrix(self.Levels[1], self.Levels[2]["node_coords"])
        L2L3Interp = interpolatePointsMatrix(self.Levels[2], self.Levels[3]["node_coords"])
        self.LInterp = [L1L2Interp, L2L3Interp]
    
    def _initialize_time_tracking(self):
        """Initialize all time tracking and state variables."""
        self.time_inc = 0
        self.record_inc = 0
        self.wait_inc = 0
        self.t_output = 0.0
        self.savenum = int(self.time_inc / self.Nonmesh["record_step"]) + 1
        
        self.initial_power = 0.0
        if "laser_power" in self.Properties:
            self.initial_power = self.Properties["laser_power"]
        
        self.laser_prev_z = float("inf")
        self._dwell_time_count = 0
        self.record_accum = True
        
        if self.record_accum:
            self.accum_time = jnp.zeros(self.Levels[0]["nn"])
            self.max_accum_time = jnp.zeros(self.Levels[0]["nn"])
        else:
            self.accum_time = None
            self.max_accum_time = None
        
        self.move_hist = [jnp.array(0), jnp.array(0), jnp.array(0)]
        
        self.force_move = False
        self.move_vert = False
        self.new_checkpoint = False
        self.ongoing_simulation = True
        self.single_step = True
    
    def _warmup_jax_compilation(self):
        """Warm-up JAX compilation to avoid GPU graph capture errors."""
        try:
            print("Warming up JAX compilation...")
            
            warmup_v = jnp.array([self.laser_start[0], self.laser_start[1], self.laser_start[2], 0.0, 0.0, 0.0, self.Properties["laser_power"]], dtype=jnp.float32)
            warmup_move_hist = [jnp.array(0, dtype=jnp.float32), jnp.array(0, dtype=jnp.float32), jnp.array(0, dtype=jnp.float32)]
            
            result = moveEverything(
                warmup_v,
                self.laser_start,
                self.Levels,
                warmup_move_hist,
                self.LInterp,
                self.L1L2Eratio,
                self.L2L3Eratio,
                self.Properties["layer_height"],
            )
            
            if isinstance(result, tuple):
                _ = [x.block_until_ready() if hasattr(x, 'block_until_ready') else None for x in result if x is not None]
            else:
                _ = result.block_until_ready() if hasattr(result, 'block_until_ready') else None
            
            if hasattr(moveEverything, '_clear_cache'):
                moveEverything._clear_cache()
            print("Warm-up complete.")
        except Exception as e:
            print(f"Warning: Warm-up failed: {e}")
            print("This may cause GPU graph capture errors. Continuing anyway...")
            import traceback
            traceback.print_exc()
    
    def _handle_layer_change(self, laser_pos):
        """
        Handle layer change logic.
        
        Parameters:
        -----------
        laser_pos : jnp.ndarray
            Current laser position
        
        Returns:
        --------
        tmp_ne_nn : tuple
            Temporary nodes and elements
        """
        if laser_pos[2] != self.laser_prev_z:
            trying_flag = True
            tmp_coords = copy.deepcopy(self.Levels[1]["orig_node_coords"])
            _L1T_state_idx = 0
            
            while trying_flag:
                if jnp.isclose(tmp_coords[2] - laser_pos[2], 0, atol=1e-4).any():
                    trying_flag = False
                else:
                    tmp_coords[2] += self.Properties["layer_height"]
                    _L1T_state_idx += 1
            
            self.Levels[1]["T0"] = jnp.maximum(
                interpolatePoints(self.Levels[1], self.Levels[1]["T0"], tmp_coords),
                self.Properties["T_amb"],
            )
            self.Levels[1]["S1_storage"] = (
                self.Levels[1]["S1_storage"]
                .at[_L1T_state_idx - 1, :]
                .set(self.Levels[1]["S1"])
            )
            self.Levels[1]["S1"] = self.Levels[1]["S1_storage"][_L1T_state_idx, :]
            self.Levels[1]["node_coords"] = copy.deepcopy(tmp_coords)
            
            L1L2Interp = interpolatePointsMatrix(self.Levels[1], self.Levels[2]["node_coords"])
            L2L3Interp = interpolatePointsMatrix(self.Levels[2], self.Levels[3]["node_coords"])
            self.LInterp = [L1L2Interp, L2L3Interp]
            tmp_ne_nn = calcStaticTmpNodesAndElements(self.Levels, laser_pos)
            
            self.laser_prev_z = laser_pos[2]
            self.force_move = True
            self.wait_inc = 0
            self.move_vert = True
            
            saveState(self.Levels[0], "Level0_", self.Nonmesh["layer_num"], self.Nonmesh["save_path"], 0)
            
            if self.record_accum:
                self.accum_time = jnp.maximum(self.accum_time, self.max_accum_time)
                jnp.savez(
                    self.Nonmesh["save_path"] + "accum_time" + str(self.Nonmesh["layer_num"]).zfill(4),
                    accum_time=self.accum_time,
                )
            
            _0nn1 = (
                self.Levels[0]["nodes"][0]
                * self.Levels[0]["nodes"][1]
                * self.Levels[0]["layer_idx_delta"]
            )
            _0nn2 = (
                self.Levels[0]["nodes"][0]
                * self.Levels[0]["nodes"][1]
                * (self.Levels[0]["nodes"][2] - self.Levels[0]["layer_idx_delta"])
            )
            
            self.Levels[0]["S1"] = self.Levels[0]["S1"].at[:_0nn2].set(self.Levels[0]["S1"][_0nn1:])
            self.Levels[0]["S1"] = self.Levels[0]["S1"].at[_0nn2:].set(0)
            
            self.Levels[0]["node_coords"][2] = (
                self.Levels[0]["orig_node_coords"][2]
                + laser_pos[2]
                - self.Levels[0]["orig_node_coords"][2][-1]
            )
            
            if self.record_accum:
                self.max_accum_time = jnp.zeros(self.Levels[0]["nn"])
                self.accum_time = self.accum_time.at[:_0nn2].set(self.accum_time[_0nn1:])
                self.accum_time = self.accum_time.at[_0nn2:].set(0)
            
            return tmp_ne_nn
        else:
            # No layer change, return current values
            tmp_ne_nn = calcStaticTmpNodesAndElements(self.Levels, laser_pos)
            return tmp_ne_nn
    
    def _finalize_simulation(self):
        """Finalize simulation: save results, calculate MSE, and cleanup."""
        saveState(self.Levels[0], "Level0_", self.Nonmesh["layer_num"], self.Nonmesh["save_path"], 0)
        
        final_power = 0.0
        if "laser_power" in self.Properties:
            final_power = self.Properties["laser_power"]
        
        saveResultsFinal(self.Levels, self.Nonmesh, power=final_power)
        
        jnp.savez(
            f"{self.Nonmesh['save_path']}FinalTemperatureFields",
            L1T=self.Levels[1]["T0"],
            L2T=self.Levels[2]["T0"],
            L3T=self.Levels[3]["T0"],
        )
        
        if self.record_accum and self.accum_time is not None and self.max_accum_time is not None:
            self.accum_time = jnp.maximum(self.accum_time, self.max_accum_time)
            jnp.savez(
                self.Nonmesh["save_path"] + "accum_time" + str(self.Nonmesh["layer_num"]).zfill(4),
                accum_time=self.accum_time,
            )
        
        try:
            stepGOMELT._clear_cache()
            stepGOMELTDwellTime._clear_cache()
            subcycleGOMELT._clear_cache()
            moveEverything._clear_cache()
            gc.collect()
        except:
            gc.collect()
        
        print("\nSimulation completed.")
    
    def nextstep(self, power: float) -> bool:
        """
        Execute one simulation step with the provided power.
        
        Parameters:
        -----------
        power : float
            Laser power to use for this step (in Watts)
        
        Returns:
        --------
        bool: True if simulation is ongoing, False if simulation is complete
        """
        # Initialize on first call
        if not self._initialized:
            self._warmup_jax_compilation()
            self._initialized = True
        
        if not self.ongoing_simulation:
            return False
        
        # Read next line from toolpath
        line = self.tool_path_file.readline()
        if not line or line.strip() == "":
            self.ongoing_simulation = False
            return False
        
        # Parse laser position from toolpath
        laser_pos_data = [float(val) for val in line.split(",")]
        laser_pos = jnp.array(laser_pos_data, dtype=jnp.float32)
        
        # Override power with provided value
        laser_pos = laser_pos.at[6].set(power)
        
        # Update wait time
        self.wait_inc = self.wait_inc + 1 if laser_pos[4] == 0 else 0
        
        # Save checkpoint if layer changes
        if laser_pos[2] != self.laser_prev_z and self.time_inc > 0:
            self.new_checkpoint = True
            try:
                stepGOMELT._clear_cache()
                stepGOMELTDwellTime._clear_cache()
                subcycleGOMELT._clear_cache()
                moveEverything._clear_cache()
                gc.collect()
            except:
                gc.collect()
        
        # Handle Layer Change
        if laser_pos[2] != self.laser_prev_z:
            tmp_ne_nn = self._handle_layer_change(laser_pos)
        else:
            tmp_ne_nn = calcStaticTmpNodesAndElements(self.Levels, laser_pos)
        
        # Move Meshes (always move for each step)
        # Clear cache before first moveEverything call
        if self.time_inc == 0 or not self._first_move:
            try:
                if hasattr(moveEverything, '_clear_cache'):
                    moveEverything._clear_cache()
                self._first_move = True
            except:
                pass
        
        (self.Levels, Shapes, self.LInterp, self.move_hist) = moveEverything(
            laser_pos,
            self.laser_start,
            self.Levels,
            self.move_hist,
            self.LInterp,
            self.L1L2Eratio,
            self.L2L3Eratio,
            self.Properties["layer_height"],
        )
        
        if self.move_vert:
            self.move_vert = False
            substrate = getSubstrateNodes(self.Levels)
            self.Levels[0]["S1"] = self.Levels[0]["S1"].at[: substrate[0]].set(1)
        else:
            substrate = getSubstrateNodes(self.Levels)
        
        self.force_move = False
        
        # Solve Thermal Fields
        if self.wait_inc <= self.Nonmesh["wait_time"]:
            # Full GO-MELT step (Levels 1–3)
            self.Levels, all_reset = stepGOMELT(
                self.Levels,
                self.ne_nn,
                tmp_ne_nn,
                Shapes,
                self.LInterp,
                laser_pos,
                self.Properties,
                laser_pos[5],  # Time step size
                power,  # Use provided power
                substrate,
            )
            
            # Update accumulated melt time
            if self.record_accum:
                _resetaccumtime = self.accum_time[self.Levels[0]["idx"]] * (all_reset > 0)
                _max_check = jnp.maximum(
                    _resetaccumtime, self.max_accum_time[self.Levels[0]["idx"]]
                )
                self.max_accum_time = self.max_accum_time.at[self.Levels[0]["idx"]].set(
                    _max_check
                )
                self.accum_time = self.accum_time.at[self.Levels[0]["idx"]].add(
                    -_resetaccumtime
                )
                
                self.accum_time = melting_temp(
                    self.Levels[3]["T0"],
                    laser_pos[5],  # Time step size
                    self.Properties["T_liquidus"],
                    self.accum_time,
                    self.Levels[0]["idx"],
                )
        else:
            # Dwell time: only update Level 1
            if (
                not (self.Levels[2]["Tprime0"] == 0).all()
                and not (self.Levels[3]["Tprime0"] == 0).all()
            ):
                self._dwell_time_count = (
                    self.Nonmesh["wait_time"] * self.Nonmesh["timestep_L3"]
                )
                self.Levels[2]["Tprime0"] = self.Levels[2]["Tprime0"].at[:].set(0)
                self.Levels[3]["Tprime0"] = self.Levels[3]["Tprime0"].at[:].set(0)
            
            self.Levels = stepGOMELTDwellTime(
                self.Levels,
                tmp_ne_nn,
                self.ne_nn,
                self.Properties,
                laser_pos[5],  # Time step size
                substrate,
            )
            self._dwell_time_count += laser_pos[5]
        
        # Increment Time and Record Counters
        self.time_inc += 1
        self.record_inc += 1
        self.t_output += laser_pos[5]
        
        # Save Checkpoint if Needed
        if self.new_checkpoint:
            self.Nonmesh["layer_num"] += 1
            FILENAME = f"Checkpoint{str(self.Nonmesh['layer_num']).zfill(4)}.pkl"
            if not os.path.exists(self.np_path):
                os.makedirs(self.np_path)
            
            save_object(
                [self.Levels, self.accum_time, self.max_accum_time, self.time_inc, self.record_inc],
                Path(self.np_path).joinpath(FILENAME),
            )
            
            # End simulation if final layer reached
            if self.Nonmesh["layer_num"] == self.layer_check:
                self.ongoing_simulation = False
                return False
            
            self.new_checkpoint = False
        
        # Save results if record step reached
        if self.record_inc >= self.Nonmesh["record_step"]:
            self.record_inc = 0
            self.savenum = int(self.time_inc / self.Nonmesh["record_step"]) + 1
            saveResults(self.Levels, self.Nonmesh, self.savenum, power=power)
        
        # Print status
        max_temp_L3 = float(jnp.max(self.Levels[3]["T0"]))
        print(f"Step {self.time_inc:6d}/{self.total_t_inc} | "
              f"Max Temp: {max_temp_L3:7.2f} K | "
              f"Power: {power:6.2f} W | "
              f"Location: X:{laser_pos[0]:6.2f} Y:{laser_pos[1]:6.2f} Z:{laser_pos[2]:6.2f}")
        
        return self.ongoing_simulation
    
    def finalize(self):
        """Finalize simulation: save results and cleanup."""
        if hasattr(self, 'tool_path_file') and self.tool_path_file:
            self.tool_path_file.close()
        self._finalize_simulation()
    
    def get_current_state(self) -> dict:
        """
        Get current simulation state (for inspection/debugging).
        
        Returns:
        --------
        dict: Dictionary with current state values (temperature, power, location, etc.)
        """
        max_temp_L3 = float(jnp.max(self.Levels[3]["T0"]))
        current_power = self.Properties["laser_power"]
        
        return {
            "time_inc": self.time_inc,
            "max_temperature": max_temp_L3,
            "current_power": current_power,
            "laser_prev_z": self.laser_prev_z,
            "ongoing_simulation": self.ongoing_simulation,
        }
    
    def get_statistics(self) -> dict:
        """
        Get simulation statistics.
        
        Returns:
        --------
        dict: Dictionary with statistics
        """
        return {
            "time_inc": self.time_inc,
            "total_t_inc": self.total_t_inc,
            "t_output": self.t_output,
        }
    
    def reset(self, solver_input: dict = None, input_file: str = None):
        """
        Reset simulator to initial state (for reuse).
        
        Parameters:
        -----------
        solver_input : dict | None
            New configuration (optional, uses existing if None)
        input_file : str | None
            New input file (optional)
        """
        # Close existing toolpath file if open
        if hasattr(self, 'tool_path_file') and self.tool_path_file:
            self.tool_path_file.close()
        
        # Update configuration if provided
        if solver_input is not None:
            self.solver_input = solver_input
        if input_file is not None:
            self.input_file = input_file
        
        # Reinitialize all components
        self._initialize_simulation_setup()
        self._setup_save_path()
        self._initialize_toolpath()
        self._initialize_interpolation()
        self._initialize_time_tracking()
        
        # Reopen toolpath file
        self.tool_path_file = open(self.Nonmesh["toolpath"], "r")
        self.np_path = Path(self.Nonmesh["save_path"] + "checkpoint").absolute()
        self.layer_check = self.Nonmesh["layer_num"] + self.Nonmesh["restart_layer_num"]
        
        # Reset flags
        self._first_move = False
        
        # Save initial results
        saveResults(self.Levels, self.Nonmesh, self.savenum, power=self.initial_power)
    
    def __del__(self):
        """Cleanup: close toolpath file if still open."""
        if hasattr(self, 'tool_path_file') and self.tool_path_file:
            try:
                self.tool_path_file.close()
            except:
                pass


def main():
    """
    Simple main function for backward compatibility.
    Initializes the class and runs the simulation.
    """
    import argparse
    
    # Clear terminal for clean output
    os.system("clear")
    
    # Parse Command-Line Arguments
    parser = argparse.ArgumentParser(
        description="GO-MELT Simulation Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python simulator.py 0 examples/example.json
        """
    )
    
    parser.add_argument(
        "device_id",
        type=int,
        nargs="?",
        default=0,
        help="GPU device ID (default: 0)"
    )
    
    parser.add_argument(
        "input_file",
        type=str,
        nargs="?",
        default="examples/example.json",
        help="Path to input JSON configuration file (default: examples/example.json)"
    )
    
    args = parser.parse_args()
    
    DEVICE_ID = args.device_id
    input_file = args.input_file
    
    # Set Environment for JAX
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    
    # Disable GPU graph capture to avoid "Failed to capture gpu graph" errors
    jax.config.update("jax_enable_x64", False)  # Use float32 for better compatibility
    
    # Disable GPU graph optimization which can cause capture errors
    try:
        jax.config.update("jax_gpu_enable_async_collectives", False)
    except:
        pass  # Older JAX versions may not have this option
    
    # Set environment variable to prevent GPU graph optimization
    if "XLA_PYTHON_CLIENT_ALLOCATOR" not in os.environ:
        os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
    
    try:
        # Attempt to assign GPU
        os.environ["CUDA_VISIBLE_DEVICES"] = str(DEVICE_ID)
    except:
        # Fallback to CPU if GPU assignment fails
        jax.config.update("jax_platform_name", "cpu")
        print("No GPU found. Running on CPU.")
    
    # Load Input File
    try:
        with open(input_file, "r") as read_file:
            solver_input = json.load(read_file)
    except FileNotFoundError:
        print(f"Error: Input file '{input_file}' not found.")
        sys.exit(1)
    
    # Launch GO-MELT Simulation
    print("Running GO-MELT")
    print(f"GPU: {DEVICE_ID}, Input File: {input_file}")
    
    # Create simulator instance
    simulator = GoMeltSimulator(solver_input, input_file)
    
    # Run simulation step by step
    while simulator.nextstep(simulator.Properties["laser_power"]):
        pass
    
    # Finalize simulation
    simulator.finalize()


if __name__ == "__main__":
    main()
