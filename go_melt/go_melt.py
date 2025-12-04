import os
import sys
import time
import argparse
import copy
import shutil
from datetime import datetime
from pathlib import Path

import dill
import jax
import jax.numpy as jnp
import numpy as np
from computeFunctions import *
from createPath import parsingGcode, count_lines
import gc
import json

# Import controllers for power control
# Ensure current directory is in path for imports
_current_dir = os.path.dirname(os.path.abspath(__file__))
if _current_dir not in sys.path:
    sys.path.insert(0, _current_dir)

try:
    from controller import RLController, PIDController
    HAS_CONTROLLERS = True
except ImportError as e:
    HAS_CONTROLLERS = False
    RLController = None
    PIDController = None
    print(f"Warning: Controllers not found ({str(e)}). Power control disabled.")


# ============================================================================
# Reusable Initialization Functions
# ============================================================================

def initialize_simulation_setup(solver_input):
    """
    Initialize simulation properties, levels, nonmesh, and compute mesh ratios.
    
    Parameters:
    -----------
    solver_input : dict
        Dictionary containing all simulation configuration
    
    Returns:
    --------
    tuple: (Properties, Levels, Nonmesh, ne_nn, subcycle, L1L2Eratio, L2L3Eratio)
    """
    Properties = SetupProperties(solver_input.get("properties", {}))
    Levels = SetupLevels(solver_input, Properties)
    Nonmesh = SetupNonmesh(solver_input.get("nonmesh", {}))
    
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
    
    return Properties, Levels, Nonmesh, ne_nn, subcycle, L1L2Eratio, L2L3Eratio


def initialize_power_controller(solver_input, Properties):
    """
    Initialize power controller (PID or RL) if configured.
    
    Parameters:
    -----------
    solver_input : dict
        Dictionary containing all simulation configuration
    Properties : dict
        Material properties dictionary
    
    Returns:
    --------
    tuple: (power_controller, controller_type)
        power_controller: Controller instance or None
        controller_type: str ("pid", "rl", or "none")
    """
    power_controller = None
    controller_type = None
    
    if HAS_CONTROLLERS:
        controller_config = solver_input.get("power_controller", {})
        controller_type = controller_config.get("type", "none").lower()
        
        if controller_type in ["pid", "rl"]:
            target_temp = controller_config.get("target_temperature", 3000.0)
            base_power = Properties["laser_power"]
            update_interval = controller_config.get("update_interval", 10)
            controller_params = controller_config.get("controller_params", {})
            
            try:
                if controller_type == "pid":
                    if PIDController is not None:
                        if "kp" not in controller_params:
                            controller_params["kp"] = 0.1
                        if "ki" not in controller_params:
                            controller_params["ki"] = 0.01
                        if "kd" not in controller_params:
                            controller_params["kd"] = 0.0
                        if "power_min" not in controller_params:
                            controller_params["power_min"] = 0.0
                        if "power_max" not in controller_params:
                            controller_params["power_max"] = 500.0
                        
                        power_controller = PIDController(
                            target_temperature=target_temp,
                            base_power=base_power,
                            update_interval=update_interval,
                            controller_params=controller_params
                        )
                        print(f"PID Controller initialized: target={target_temp}K, base_power={base_power}W")
                        print(f"  PID gains: kp={controller_params['kp']}, ki={controller_params['ki']}, kd={controller_params['kd']}")
                    else:
                        print("Warning: PIDController not available. Power control disabled.")
                
                elif controller_type == "rl":
                    if RLController is not None:
                        if "kp" not in controller_params:
                            controller_params["kp"] = 0.1
                        if "power_min" not in controller_params:
                            controller_params["power_min"] = 0.0
                        if "power_max" not in controller_params:
                            controller_params["power_max"] = 500.0
                        
                        rl_model_path = controller_params.get("rl_model_path", None)
                        
                        power_controller = RLController(
                            target_temperature=target_temp,
                            base_power=base_power,
                            update_interval=update_interval,
                            controller_params=controller_params,
                            rl_model_path=rl_model_path
                        )
                        if rl_model_path and power_controller.use_rl:
                            print(f"RL Controller initialized with trained model: target={target_temp}K, base_power={base_power}W")
                        else:
                            print(f"RL Controller initialized (linear): target={target_temp}K, base_power={base_power}W")
                        
                        power_controller._print_first_observation = True
                    else:
                        print("Warning: RLController (RL) not available. Power control disabled.")
                
            except Exception as e:
                print(f"Warning: Failed to initialize {controller_type.upper()} controller: {e}")
                power_controller = None
        else:
            if controller_type != "none":
                print(f"Warning: Unknown controller type '{controller_type}'. Valid options: 'pid', 'rl', 'none'")
            print("Running without power controller (using toolpath power).")
    
    return power_controller, controller_type


def setup_save_path(Nonmesh, input_file, controller_type):
    """
    Setup save path with timestamp and update Nonmesh.
    
    Parameters:
    -----------
    Nonmesh : dict
        Nonmesh configuration dictionary
    input_file : str, optional
        Path to the input JSON file
    controller_type : str
        Controller type ("pid", "rl", or "none")
    
    Returns:
    --------
    Nonmesh : dict
        Updated Nonmesh with new save_path and toolpath
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if input_file:
        json_name = Path(input_file).stem
    else:
        json_name = controller_type if controller_type else "none"
    
    base_save_path = Nonmesh.get("save_path", "./results/")
    base_path = Path(base_save_path.rstrip("/")).parent
    new_save_path = base_path / f"{timestamp}_{json_name}"
    
    new_save_path.mkdir(parents=True, exist_ok=True)
    new_save_path_str = str(new_save_path) + "/"
    
    Nonmesh["save_path"] = new_save_path_str
    
    original_toolpath = Nonmesh.get("toolpath", "./results/example/toolpath.txt")
    toolpath_filename = Path(original_toolpath).name
    new_toolpath = new_save_path / toolpath_filename
    
    if Nonmesh.get("use_txt", 0):
        original_toolpath_path = Path(original_toolpath)
        if original_toolpath_path.exists():
            shutil.copy2(original_toolpath_path, new_toolpath)
            print(f"Copied toolpath file to: {new_toolpath}")
        else:
            print(f"Warning: Toolpath file not found at {original_toolpath}")
    
    Nonmesh["toolpath"] = str(new_toolpath)
    
    print(f"Results will be saved to: {new_save_path_str}")
    
    return Nonmesh


def initialize_toolpath(Nonmesh, Properties, Levels):
    """
    Parse toolpath and get initial laser position.
    
    Parameters:
    -----------
    Nonmesh : dict
        Nonmesh configuration dictionary
    Properties : dict
        Material properties dictionary
    Levels : dict
        Levels dictionary
    
    Returns:
    --------
    tuple: (total_t_inc, laser_start)
        total_t_inc: int, total time steps
        laser_start: np.ndarray, initial laser position
    """
    if Nonmesh["use_txt"]:
        move_mesh = count_lines(Nonmesh["toolpath"])
    else:
        move_mesh = parsingGcode(Nonmesh, Properties, Levels[2]["h"])
    
    total_t_inc = move_mesh
    
    if not Properties["laser_center"]:
        with open(Nonmesh["toolpath"], "r") as tool_path_file:
            laser_start = np.array(
                [float(val) for val in tool_path_file.readline().split(",")]
            )
    else:
        laser_start = np.array(Properties["laser_center"])
    
    return total_t_inc, laser_start


def initialize_interpolation(Levels):
    """
    Initialize interpolation matrices between levels.
    
    Parameters:
    -----------
    Levels : dict
        Levels dictionary
    
    Returns:
    --------
    LInterp : list
        List of interpolation matrices [L1L2Interp, L2L3Interp]
    """
    L1L2Interp = interpolatePointsMatrix(Levels[1], Levels[2]["node_coords"])
    L2L3Interp = interpolatePointsMatrix(Levels[2], Levels[3]["node_coords"])
    LInterp = [L1L2Interp, L2L3Interp]
    
    return LInterp


def initialize_time_tracking(Levels, Nonmesh, power_controller, Properties=None):
    """
    Initialize time tracking, error tracking, and layer tracking variables.
    
    Parameters:
    -----------
    Levels : dict
        Levels dictionary
    Nonmesh : dict
        Nonmesh configuration dictionary
    power_controller : object or None
        Power controller instance
    Properties : dict, optional
        Material properties dictionary (for getting initial power)
    
    Returns:
    --------
    dict: Dictionary containing all tracking variables
    """
    time_inc = record_inc = wait_inc = 0
    t_output = 0.0
    savenum = int(time_inc / Nonmesh["record_step"]) + 1
    
    initial_power = 0.0
    if Properties is not None and "laser_power" in Properties:
        initial_power = Properties["laser_power"]
    
    temperature_errors = []
    target_temperature = None
    if power_controller is not None:
        target_temperature = power_controller.target_temperature
        initial_power = power_controller.get_current_power()
    
    laser_prev_z = float("inf")
    _dwell_time_count = 0
    record_accum = True
    
    if record_accum:
        accum_time = jnp.zeros(Levels[0]["nn"])
        max_accum_time = jnp.zeros(Levels[0]["nn"])
    else:
        accum_time = None
        max_accum_time = None
    
    move_hist = [jnp.array(0), jnp.array(0), jnp.array(0)]
    
    force_move = move_vert = new_checkpoint = False
    ongoing_simulation = single_step = True
    
    return {
        "time_inc": time_inc,
        "record_inc": record_inc,
        "wait_inc": wait_inc,
        "t_output": t_output,
        "savenum": savenum,
        "temperature_errors": temperature_errors,
        "target_temperature": target_temperature,
        "laser_prev_z": laser_prev_z,
        "_dwell_time_count": _dwell_time_count,
        "record_accum": record_accum,
        "accum_time": accum_time,
        "max_accum_time": max_accum_time,
        "move_hist": move_hist,
        "force_move": force_move,
        "move_vert": move_vert,
        "new_checkpoint": new_checkpoint,
        "ongoing_simulation": ongoing_simulation,
        "single_step": single_step,
        "initial_power": initial_power
    }


def load_checkpoint(Nonmesh, Levels, tool_path_file):
    """
    Load checkpoint if requested.
    
    Parameters:
    -----------
    Nonmesh : dict
        Nonmesh configuration dictionary
    Levels : dict
        Levels dictionary (will be updated)
    tool_path_file : file object
        Open toolpath file
    
    Returns:
    --------
    tuple: (load_chkpt, time_inc_loaded, record_inc)
        load_chkpt: bool, whether checkpoint was loaded
        time_inc_loaded: int, time increment from checkpoint
        record_inc: int, record increment from checkpoint
    """
    if Nonmesh["layer_num"] > 0:
        print(f"Loading checkpoint: Layer {Nonmesh['layer_num']}")
        np_path = Path(Nonmesh["save_path"] + "checkpoint").absolute()
        FILENAME = f"Checkpoint{str(Nonmesh['layer_num']).zfill(4)}.pkl"
        
        with open(np_path.joinpath(FILENAME), "rb") as f:
            Levels, accum_time, max_accum_time, time_inc_loaded, record_inc = dill.load(f)
        
        load_chkpt = True
        line_len = len(tool_path_file.readline())
        tool_path_file.seek(time_inc_loaded * line_len)
        
        return load_chkpt, time_inc_loaded, record_inc, accum_time, max_accum_time
    else:
        return False, 0, 0, None, None


def warmup_jax_compilation(laser_start, Levels, LInterp, L1L2Eratio, L2L3Eratio, Properties, load_chkpt):
    """
    Warm-up JAX compilation to avoid GPU graph capture errors.
    
    Parameters:
    -----------
    laser_start : np.ndarray
        Initial laser position
    Levels : dict
        Levels dictionary
    LInterp : list
        Interpolation matrices
    L1L2Eratio : list
        Level 1 to Level 2 element ratios
    L2L3Eratio : list
        Level 2 to Level 3 element ratios
    Properties : dict
        Material properties dictionary
    load_chkpt : bool
        Whether checkpoint was loaded (skip warm-up if True)
    """
    if not load_chkpt:
        try:
            print("Warming up JAX compilation...")
            from computeFunctions import moveEverything
            
            warmup_v = jnp.array([laser_start[0], laser_start[1], laser_start[2], 0.0, 0.0, 0.0, Properties["laser_power"]], dtype=jnp.float32)
            warmup_move_hist = [jnp.array(0, dtype=jnp.float32), jnp.array(0, dtype=jnp.float32), jnp.array(0, dtype=jnp.float32)]
            
            result = moveEverything(
                warmup_v,
                laser_start,
                Levels,
                warmup_move_hist,
                LInterp,
                L1L2Eratio,
                L2L3Eratio,
                Properties["layer_height"],
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


def handle_layer_change(laser_pos, laser_prev_z, Levels, Properties, LInterp, Nonmesh, 
                       load_chkpt, record_accum, accum_time, max_accum_time):
    """
    Handle layer change logic.
    
    Parameters:
    -----------
    laser_pos : jnp.ndarray
        Current laser position
    laser_prev_z : float
        Previous z-coordinate
    Levels : dict
        Levels dictionary (will be updated)
    Properties : dict
        Material properties dictionary
    LInterp : list
        Interpolation matrices (will be updated)
    Nonmesh : dict
        Nonmesh configuration dictionary
    load_chkpt : bool
        Whether loading from checkpoint
    record_accum : bool
        Whether to record accumulation
    accum_time : jnp.ndarray
        Accumulated time array
    max_accum_time : jnp.ndarray
        Maximum accumulated time array
    
    Returns:
    --------
    tuple: (LInterp, tmp_ne_nn, laser_prev_z, force_move, move_vert, wait_inc, accum_time, max_accum_time)
    """
    if laser_pos[2] != laser_prev_z:
        trying_flag = True
        tmp_coords = copy.deepcopy(Levels[1]["orig_node_coords"])
        _L1T_state_idx = 0
        
        while trying_flag:
            if jnp.isclose(tmp_coords[2] - laser_pos[2], 0, atol=1e-4).any():
                trying_flag = False
            else:
                tmp_coords[2] += Properties["layer_height"]
                _L1T_state_idx += 1
        
        if not load_chkpt:
            Levels[1]["T0"] = jnp.maximum(
                interpolatePoints(Levels[1], Levels[1]["T0"], tmp_coords),
                Properties["T_amb"],
            )
            Levels[1]["S1_storage"] = (
                Levels[1]["S1_storage"]
                .at[_L1T_state_idx - 1, :]
                .set(Levels[1]["S1"])
            )
            Levels[1]["S1"] = Levels[1]["S1_storage"][_L1T_state_idx, :]
            Levels[1]["node_coords"] = copy.deepcopy(tmp_coords)
        
        L1L2Interp = interpolatePointsMatrix(Levels[1], Levels[2]["node_coords"])
        L2L3Interp = interpolatePointsMatrix(Levels[2], Levels[3]["node_coords"])
        LInterp = [L1L2Interp, L2L3Interp]
        tmp_ne_nn = calcStaticTmpNodesAndElements(Levels, laser_pos)
        
        laser_prev_z = laser_pos[2]
        force_move = True
        wait_inc = 0
        move_vert = True
        
        if not load_chkpt:
            saveState(Levels[0], "Level0_", Nonmesh["layer_num"], Nonmesh["save_path"], 0)
            
            if record_accum:
                accum_time = jnp.maximum(accum_time, max_accum_time)
                jnp.savez(
                    Nonmesh["save_path"] + "accum_time" + str(Nonmesh["layer_num"]).zfill(4),
                    accum_time=accum_time,
                )
            
            _0nn1 = (
                Levels[0]["nodes"][0]
                * Levels[0]["nodes"][1]
                * Levels[0]["layer_idx_delta"]
            )
            _0nn2 = (
                Levels[0]["nodes"][0]
                * Levels[0]["nodes"][1]
                * (Levels[0]["nodes"][2] - Levels[0]["layer_idx_delta"])
            )
            
            Levels[0]["S1"] = Levels[0]["S1"].at[:_0nn2].set(Levels[0]["S1"][_0nn1:])
            Levels[0]["S1"] = Levels[0]["S1"].at[_0nn2:].set(0)
            
            Levels[0]["node_coords"][2] = (
                Levels[0]["orig_node_coords"][2]
                + laser_pos[2]
                - Levels[0]["orig_node_coords"][2][-1]
            )
            
            if record_accum:
                max_accum_time = jnp.zeros(Levels[0]["nn"])
                accum_time = accum_time.at[:_0nn2].set(accum_time[_0nn1:])
                accum_time = accum_time.at[_0nn2:].set(0)
        
        return LInterp, tmp_ne_nn, laser_prev_z, force_move, move_vert, wait_inc, accum_time, max_accum_time
    else:
        # No layer change, return current values
        tmp_ne_nn = calcStaticTmpNodesAndElements(Levels, laser_pos)
        return LInterp, tmp_ne_nn, laser_prev_z, False, False, None, accum_time, max_accum_time


def finalize_simulation(Levels, Nonmesh, power_controller, target_temperature, 
                       temperature_errors, controller_type, record_accum, accum_time, max_accum_time, Properties=None):
    """
    Finalize simulation: save results, calculate MSE, and cleanup.
    
    Parameters:
    -----------
    Levels : dict
        Levels dictionary
    Nonmesh : dict
        Nonmesh configuration dictionary
    power_controller : object or None
        Power controller instance
    target_temperature : float or None
        Target temperature for MSE calculation
    temperature_errors : list
        List of squared temperature errors
    controller_type : str
        Controller type
    record_accum : bool
        Whether accumulation was recorded
    accum_time : jnp.ndarray or None
        Accumulated time array
    max_accum_time : jnp.ndarray or None
        Maximum accumulated time array
    Properties : dict, optional
        Material properties dictionary (for getting final power)
    """
    saveState(Levels[0], "Level0_", Nonmesh["layer_num"], Nonmesh["save_path"], 0)
    
    final_power = 0.0
    if Properties is not None and "laser_power" in Properties:
        final_power = Properties["laser_power"]
    if power_controller is not None:
        final_power = power_controller.get_current_power()
    
    saveResultsFinal(Levels, Nonmesh, power=final_power)
    
    jnp.savez(
        f"{Nonmesh['save_path']}FinalTemperatureFields",
        L1T=Levels[1]["T0"],
        L2T=Levels[2]["T0"],
        L3T=Levels[3]["T0"],
    )
    
    if record_accum and accum_time is not None and max_accum_time is not None:
        accum_time = jnp.maximum(accum_time, max_accum_time)
        jnp.savez(
            Nonmesh["save_path"] + "accum_time" + str(Nonmesh["layer_num"]).zfill(4),
            accum_time=accum_time,
        )
    
    if target_temperature is not None and len(temperature_errors) > 0:
        mse = np.mean(temperature_errors)
        rmse = np.sqrt(mse)
        print(f"\n{'='*60}")
        print(f"Temperature Control Statistics:")
        print(f"  Target Temperature: {target_temperature:.2f} K")
        print(f"  Mean Squared Error (MSE): {mse:.2f} K²")
        print(f"  Root Mean Squared Error (RMSE): {rmse:.2f} K")
        print(f"  Number of samples: {len(temperature_errors)}")
        print(f"{'='*60}\n")
        
        mse_data = {
            "target_temperature": float(target_temperature),
            "mse": float(mse),
            "rmse": float(rmse),
            "num_samples": len(temperature_errors),
            "controller_type": controller_type if controller_type else "none"
        }
        mse_file = Path(Nonmesh["save_path"]) / "mse_stats.json"
        with open(mse_file, "w") as f:
            json.dump(mse_data, f, indent=2)
        print(f"MSE statistics saved to: {mse_file}")
    else:
        print("\nNo temperature control active - MSE not calculated.\n")
    
    try:
        stepGOMELT._clear_cache()
        stepGOMELTDwellTime._clear_cache()
        subcycleGOMELT._clear_cache()
        moveEverything._clear_cache()
        gc.collect()
    except:
        gc.collect()
    
    print("\nSimulation completed.")


def go_melt(solver_input: dict, input_file: str = None):
    """
    Main GO-MELT simulation driver. This function initializes the simulation,
    sets up all levels, properties, and toolpath data, and prepares for time stepping.
    Thermal solves using the GO-MELT algorithm are then used.
    
    Parameters:
    -----------
    solver_input : dict
        Dictionary containing all simulation configuration
    input_file : str, optional
        Path to the input JSON file (used for naming output directory)
    """
    tstart = time.time()  # Start timer
    
    # Initialize flag for first moveEverything call
    go_melt._first_move = False

    # -------------------------------
    # Initialize Simulation Setup
    # -------------------------------
    Properties, Levels, Nonmesh, ne_nn, subcycle, L1L2Eratio, L2L3Eratio = initialize_simulation_setup(solver_input)

    # -------------------------------
    # Initialize Power Controller
    # -------------------------------
    power_controller, controller_type = initialize_power_controller(solver_input, Properties)

    # -------------------------------
    # Setup Save Path
    # -------------------------------
    Nonmesh = setup_save_path(Nonmesh, input_file, controller_type)

    # -------------------------------
    # Initialize Toolpath
    # -------------------------------
    total_t_inc, laser_start = initialize_toolpath(Nonmesh, Properties, Levels)

    # -------------------------------
    # Initialize Interpolation
    # -------------------------------
    LInterp = initialize_interpolation(Levels)

    # -------------------------------
    # Initialize Time Tracking
    # -------------------------------
    tracking = initialize_time_tracking(Levels, Nonmesh, power_controller, Properties)
    time_inc = tracking["time_inc"]
    record_inc = tracking["record_inc"]
    wait_inc = tracking["wait_inc"]
    t_output = tracking["t_output"]
    savenum = tracking["savenum"]
    temperature_errors = tracking["temperature_errors"]
    target_temperature = tracking["target_temperature"]
    laser_prev_z = tracking["laser_prev_z"]
    _dwell_time_count = tracking["_dwell_time_count"]
    record_accum = tracking["record_accum"]
    accum_time = tracking["accum_time"]
    max_accum_time = tracking["max_accum_time"]
    move_hist = tracking["move_hist"]
    force_move = tracking["force_move"]
    move_vert = tracking["move_vert"]
    new_checkpoint = tracking["new_checkpoint"]
    ongoing_simulation = tracking["ongoing_simulation"]
    single_step = tracking["single_step"]
    initial_power = tracking["initial_power"]
    
    # Save initial results
    saveResults(Levels, Nonmesh, savenum, power=initial_power)

    # -------------------------------
    # Toolpath File & Checkpointing
    # -------------------------------
    tool_path_file = open(Nonmesh["toolpath"], "r")
    np_path = Path(Nonmesh["save_path"] + "checkpoint").absolute()
    layer_check = Nonmesh["layer_num"] + Nonmesh["restart_layer_num"]

    # -------------------------------
    # Load Checkpoint if Requested
    # -------------------------------
    load_chkpt, time_inc_loaded, record_inc_loaded, accum_time_loaded, max_accum_time_loaded = load_checkpoint(
        Nonmesh, Levels, tool_path_file
    )
    if load_chkpt:
        time_inc += time_inc_loaded
        record_inc = record_inc_loaded
        if accum_time_loaded is not None:
            accum_time = accum_time_loaded
        if max_accum_time_loaded is not None:
            max_accum_time = max_accum_time_loaded

    # -------------------------------
    # Warm-up JAX Compilation
    # -------------------------------
    warmup_jax_compilation(laser_start, Levels, LInterp, L1L2Eratio, L2L3Eratio, Properties, load_chkpt)

    # -------------------------------
    # Start Time Loop
    # -------------------------------
    while ongoing_simulation:
        t_loop = time.time()  # Start timer for this loop

        # -----------------------------------
        # Read laser path for one full subcycle (n2 × n3 lines)
        # -----------------------------------
        _pos = [
            tool_path_file.readline().split(",")
            for _2 in range(subcycle[0] * subcycle[1])
        ]

        # Convert laser path to array if valid, else handle end-of-file
        if [""] not in _pos:
            t_add = subcycle[2]
            laser_all = jnp.array([[float(val) for val in line] for line in _pos])
        else:
            # Set up for partial single time-stepping and end simulation afterwards
            t_add = _pos.index([""])
            laser_all = jnp.array(
                [[float(val) for val in _pos[i]] for i in range(t_add)]
            )
            ongoing_simulation = False
            if t_add == 0:
                break

        # -----------------------------------
        # Determine if single-step is needed
        # -----------------------------------
        single_step = (
            any(laser_all[:, 2] != laser_prev_z)
            or load_chkpt
            or (wait_inc > max(0, Nonmesh["wait_time"] - subcycle[0] * subcycle[1] * 2))
            or not ongoing_simulation
            or laser_all.shape[0] == 1
            or (
                jnp.abs(jnp.diff(laser_all, axis=0)[:, :2] / laser_all[:-1, 5].max())
                > (100 * Nonmesh["laser_velocity"])
            ).any()
        )

        # -----------------------------------
        # Single-Step Execution (Equal time step for each Level)
        # -----------------------------------
        if single_step:
            # Initialize substrate before first use
            substrate = getSubstrateNodes(Levels)
            
            for laser_pos in laser_all:
                wait_inc = wait_inc + 1 if laser_pos[4] == 0 else 0

                # Save checkpoint if layer changes and not from checkpoint
                if laser_pos[2] != laser_prev_z and time_inc > 0 and not load_chkpt:
                    new_checkpoint = True
                    try:
                        stepGOMELT._clear_cache()
                        stepGOMELTDwellTime._clear_cache()
                        subcycleGOMELT._clear_cache()
                        moveEverything._clear_cache()
                        gc.collect()
                        # Cache cleared
                    except:
                        gc.collect()

                # -----------------------------------
                # Handle Layer Change
                # -----------------------------------
                if laser_pos[2] != laser_prev_z:
                    LInterp, tmp_ne_nn, laser_prev_z, force_move, move_vert, wait_inc_new, accum_time, max_accum_time = handle_layer_change(
                        laser_pos, laser_prev_z, Levels, Properties, LInterp, Nonmesh,
                        load_chkpt, record_accum, accum_time, max_accum_time
                    )
                    if wait_inc_new is not None:
                        wait_inc = wait_inc_new
                else:
                    tmp_ne_nn = calcStaticTmpNodesAndElements(Levels, laser_pos)

                force_move = True

                # -----------------------------------
                # Move Meshes if Needed
                # -----------------------------------
                if force_move:
                    force_move = False
                    # Clear cache before first moveEverything call to avoid GPU graph capture issues
                    # This is especially important for PID controller which may trigger different code paths
                    if time_inc == 0 or not hasattr(go_melt, '_first_move'):
                        try:
                            from computeFunctions import moveEverything
                            if hasattr(moveEverything, '_clear_cache'):
                                moveEverything._clear_cache()
                            go_melt._first_move = True
                        except:
                            pass
                    
                    # Ensure laser_pos is float32 for consistency
                    if isinstance(laser_pos, jnp.ndarray):
                        laser_pos = jnp.asarray(laser_pos, dtype=jnp.float32)
                    elif isinstance(laser_pos, (list, tuple, np.ndarray)):
                        laser_pos = jnp.array(laser_pos, dtype=jnp.float32)
                    
                    (Levels, Shapes, LInterp, move_hist) = moveEverything(
                        laser_pos,
                        laser_start,
                        Levels,
                        move_hist,
                        LInterp,
                        L1L2Eratio,
                        L2L3Eratio,
                        Properties["layer_height"],
                    )

                    if move_vert:
                        move_vert = False
                        substrate = getSubstrateNodes(Levels)
                        Levels[0]["S1"] = Levels[0]["S1"].at[: substrate[0]].set(1)
                        # New layer started

                # -----------------------------------
                # RL Power Control (get controlled power for current step)
                # -----------------------------------
                # Determine power to use: controlled power if available, otherwise toolpath power
                current_power = float(laser_pos[6])  # Default: power from toolpath
                if power_controller is not None and wait_inc <= Nonmesh["wait_time"]:
                    current_power = power_controller.get_current_power()

                # -----------------------------------
                # Solve Thermal Fields
                # -----------------------------------
                if wait_inc <= Nonmesh["wait_time"]:
                    # Full GO-MELT step (Levels 1–3)
                    # Use controlled power if available, otherwise use toolpath power
                    Levels, all_reset = stepGOMELT(
                        Levels,
                        ne_nn,
                        tmp_ne_nn,
                        Shapes,
                        LInterp,
                        laser_pos,
                        Properties,
                        laser_pos[5],  # Time step size
                        current_power,  # Use controlled power
                        substrate,
                    )

                    # Temperature/power/location already printed in main loop

                    # -----------------------------------
                    # RL Power Control (after thermal solve - update for next steps)
                    # -----------------------------------
                    if power_controller is not None:
                        try:
                            # Execute controller step with updated Levels (has current temperature)
                            new_power, power_updated = power_controller.step(
                                Levels,
                                laser_all=laser_all if 'laser_all' in locals() else None,
                                accum_time=accum_time if 'accum_time' in locals() else None,
                                accum_idx=Levels[0]["idx"] if 'accum_time' in locals() else None,
                                Properties=Properties
                            )
                            
                            if power_updated:
                                # Get current temperature from history
                                max_temp = power_controller.temperature_history[-1] if power_controller.temperature_history else 0.0
                                print(f"  -> Power updated: {new_power:.2f} W, Temp: {max_temp:.2f} K (target: {power_controller.target_temperature:.2f} K)")
                        except Exception as e:
                            # If controller fails, continue with current power
                            print(f"Warning: Power controller error: {e}. Continuing with current power.")

                    # Update accumulated melt time
                    if record_accum:
                        _resetaccumtime = accum_time[Levels[0]["idx"]] * (all_reset > 0)
                        _max_check = jnp.maximum(
                            _resetaccumtime, max_accum_time[Levels[0]["idx"]]
                        )
                        max_accum_time = max_accum_time.at[Levels[0]["idx"]].set(
                            _max_check
                        )
                        accum_time = accum_time.at[Levels[0]["idx"]].add(
                            -_resetaccumtime
                        )

                        accum_time = melting_temp(
                            Levels[3]["T0"],
                            laser_pos[5],  # Time step size
                            Properties["T_liquidus"],
                            accum_time,
                            Levels[0]["idx"],
                        )
                else:
                    # Dwell time: only update Level 1
                    if (
                        not (Levels[2]["Tprime0"] == 0).all()
                        and not (Levels[3]["Tprime0"] == 0).all()
                    ):
                        _dwell_time_count = (
                            Nonmesh["wait_time"] * Nonmesh["timestep_L3"]
                        )
                        Levels[2]["Tprime0"] = Levels[2]["Tprime0"].at[:].set(0)
                        Levels[3]["Tprime0"] = Levels[3]["Tprime0"].at[:].set(0)

                    Levels = stepGOMELTDwellTime(
                        Levels,
                        tmp_ne_nn,
                        ne_nn,
                        Properties,
                        laser_pos[5],  # Time step size
                        substrate,
                    )
                    _dwell_time_count += laser_pos[5]
                    # Dwell time: {_dwell_time_count:.6f} s

                # -----------------------------------
                # Increment Time and Record Counters
                # -----------------------------------
                time_inc += 1
                record_inc += 1

            # -----------------------------------
            # Save Checkpoint if Needed
            # -----------------------------------
            if new_checkpoint:
                Nonmesh["layer_num"] += 1
                # Saving checkpoint for Layer {Nonmesh['layer_num']}
                FILENAME = f"Checkpoint{str(Nonmesh['layer_num']).zfill(4)}.pkl"
                if not os.path.exists(np_path):
                    os.makedirs(np_path)

                save_object(
                    [Levels, accum_time, max_accum_time, time_inc, record_inc],
                    Path(np_path).joinpath(FILENAME),
                )
                # Checkpoint saved

                # End simulation if final layer reached
                if Nonmesh["layer_num"] == layer_check:
                    return

                new_checkpoint = False
                load_chkpt = True  # Use single-step for next layer
            else:
                load_chkpt = False

        else:  # Subcycling mode
            # Initialize substrate and tmp_ne_nn before use
            substrate = getSubstrateNodes(Levels)
            tmp_ne_nn = calcStaticTmpNodesAndElements(Levels, laser_all[0, :])
            
            # Update wait time if laser is off
            wait_inc = (
                wait_inc + len(laser_all) - laser_all[:, 4].sum()
                if (laser_all[:, 4] == 0).any()
                else 0
            )

            # Always move mesh in subcycling
            (Levels, Shapes, LInterp, move_hist) = moveEverything(
                laser_all[0, :],
                laser_start,
                Levels,
                move_hist,
                LInterp,
                L1L2Eratio,
                L2L3Eratio,
                Properties["layer_height"],
            )

            # Extract power values for each substep
            _P = laser_all[:, 6]
            
            # RL Power Control: Update power array for subcycling if controller is active
            if power_controller is not None:
                current_power = power_controller.get_current_power()
                # Override all powers in this subcycle with controlled power
                _P = jnp.array([float(current_power)] * len(_P))

            # Run full GO-MELT subcycling
            Levels, L2all, L3all, L3pall, _max_accum, _accum = subcycleGOMELT(
                Levels,
                ne_nn,
                Shapes,
                substrate,
                LInterp,
                tmp_ne_nn,
                laser_all,
                Properties,
                _P,
                subcycle,
                max_accum_time[Levels[0]["idx"]],
                accum_time[Levels[0]["idx"]],
            )
            gc.collect()

            if record_accum:
                max_accum_time = max_accum_time.at[Levels[0]["idx"]].set(_max_accum)
                accum_time = accum_time.at[Levels[0]["idx"]].set(_accum)

            # RL Power Control: Update power after subcycling (for next cycle)
            if power_controller is not None:
                try:
                    # Execute controller step with updated Levels
                    new_power, power_updated = power_controller.step(
                        Levels,
                        laser_all=laser_all,
                        accum_time=accum_time[Levels[0]["idx"]] if 'accum_time' in locals() else None,
                        accum_idx=Levels[0]["idx"],
                        Properties=Properties
                    )
                    
                    if power_updated:
                        # Get current temperature from history
                        max_temp = power_controller.temperature_history[-1] if power_controller.temperature_history else 0.0
                        print(f"  -> Power updated: {new_power:.2f} W, Temp: {max_temp:.2f} K (target: {power_controller.target_temperature:.2f} K)")
                except Exception as e:
                    print(f"Warning: Power controller error in subcycling: {e}")

            # Update counters and total elapsed time
            time_inc += t_add
            record_inc += t_add

        # -----------------------------------
        # Output and Monitoring
        # -----------------------------------
        t_output += laser_all[:, 5].sum()

        # Save results if record step reached
        if record_inc >= Nonmesh["record_step"]:
            record_inc = 0
            savenum = int(time_inc / Nonmesh["record_step"]) + 1
            # Get current power for saving
            current_power_for_save = Properties["laser_power"]
            if power_controller is not None:
                current_power_for_save = power_controller.get_current_power()
            elif single_step and len(laser_all) > 0:
                # Use power from last laser position in single-step mode
                current_power_for_save = float(laser_all[-1, 6])
            elif not single_step and len(laser_all) > 0:
                # Use power from last laser position in subcycling mode
                current_power_for_save = float(laser_all[-1, 6])
            saveResults(Levels, Nonmesh, savenum, power=current_power_for_save)

        # Simple printing: Focus on temps, power, location
        # Get current temperature (max from Level 3)
        max_temp_L3 = float(jnp.max(Levels[3]["T0"]))
        
        # Track temperature error for MSE calculation
        if target_temperature is not None:
            error = target_temperature - max_temp_L3
            temperature_errors.append(error * error)  # Store squared error
        
        # Get current power
        current_power_display = Properties["laser_power"]
        if power_controller is not None:
            current_power_display = power_controller.get_current_power()
        elif len(laser_all) > 0:
            current_power_display = float(laser_all[-1, 6])
        
        # Get current location
        current_loc = laser_all[-1, :] if len(laser_all) > 0 else jnp.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, current_power_display])
        
        # Simple print format: Step | Max Temp | Power | Location
        print(f"Step {time_inc:6d}/{total_t_inc} | "
              f"Max Temp: {max_temp_L3:7.2f} K | "
              f"Power: {current_power_display:6.2f} W | "
              f"Location: X:{current_loc[0]:6.2f} Y:{current_loc[1]:6.2f} Z:{current_loc[2]:6.2f}")

    # -----------------------------------
    # Finalization
    # -----------------------------------
    tool_path_file.close()
    
    finalize_simulation(
        Levels, Nonmesh, power_controller, target_temperature,
        temperature_errors, controller_type, record_accum, accum_time, max_accum_time, Properties
    )


if __name__ == "__main__":
    # Clear terminal for clean output
    os.system("clear")

    # -------------------------------
    # Parse Command-Line Arguments
    # -------------------------------
    parser = argparse.ArgumentParser(
        description="GO-MELT Simulation Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python go_melt.py 0 examples/rl.json
  python go_melt.py 0 examples/rl.json --checkpoints-dir ./rl_training/checkpoints
  python go_melt.py 0 examples/rl.json --checkpoints-dir ./rl_training/checkpoints --checkpoint-name best_model
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
    
    parser.add_argument(
        "--checkpoints-dir",
        type=str,
        default=None,
        help="Directory containing RL model checkpoints (e.g., ./rl_training/checkpoints)"
    )
    
    parser.add_argument(
        "--checkpoint-name",
        type=str,
        default="best_model",
        help="Name of checkpoint to use (without .zip extension, default: best_model). "
             "Will look for {name}.zip in checkpoints directory"
    )
    
    args = parser.parse_args()
    
    DEVICE_ID = args.device_id
    input_file = args.input_file
    
    # Check for checkpoints folder if specified
    checkpoints_dir = args.checkpoints_dir
    checkpoint_name = args.checkpoint_name
    
    if checkpoints_dir:
        checkpoints_path = Path(checkpoints_dir)
        if not checkpoints_path.exists():
            print(f"Warning: Checkpoints directory not found: {checkpoints_dir}")
            print("Continuing without checkpoint override.")
            checkpoints_dir = None
        else:
            # Look for the specified checkpoint
            checkpoint_file = checkpoints_path / f"{checkpoint_name}.zip"
            if not checkpoint_file.exists():
                print(f"Warning: Checkpoint file not found: {checkpoint_file}")
                print("Available files in checkpoints directory:")
                for f in checkpoints_path.glob("*.zip"):
                    print(f"  - {f.name}")
                print("Continuing without checkpoint override.")
                checkpoints_dir = None

    # -------------------------------
    # Set Environment for JAX
    # -------------------------------
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    
    # Disable GPU graph capture to avoid "Failed to capture gpu graph" errors
    # This is especially important for PID controller which may trigger different code paths
    # GPU graph capture fails when code paths differ between calls (e.g., with/without controller)
    import jax
    jax.config.update("jax_enable_x64", False)  # Use float32 for better compatibility
    
    # Disable GPU graph optimization which can cause capture errors
    # Note: Some JAX versions may not support all these options
    try:
        jax.config.update("jax_gpu_enable_async_collectives", False)
    except:
        pass  # Older JAX versions may not have this option
    
    # Disable GPU graph capture entirely to avoid "Failed to capture gpu graph" errors
    # This is especially important when using controllers (PID/RL) that may change code paths
    # Note: We don't set XLA_FLAGS with invalid flags - only use JAX config methods
    try:
        # Method 1: Disable via JAX config if available
        try:
            jax.config.update("jax_gpu_enable_async_collectives", False)
        except:
            pass
        
        # Method 2: Try to disable GPU graph capture via environment (if supported)
        # Only set valid XLA_FLAGS - don't add invalid flags
        # The XLA_FLAGS environment variable should be set by user if needed
    except Exception as e:
        print(f"Warning: Could not fully disable GPU graph capture: {e}")
    
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

    # -------------------------------
    # Load Input File
    # -------------------------------
    try:
        with open(input_file, "r") as read_file:
            solver_input = json.load(read_file)
    except FileNotFoundError:
        print(f"Error: Input file '{input_file}' not found.")
        sys.exit(1)
    
    # Update checkpoint path if checkpoints directory was specified
    if checkpoints_dir:
        checkpoint_file = Path(checkpoints_dir) / f"{checkpoint_name}.zip"
        if checkpoint_file.exists():
            # Update the RL model path in the config
            if "power_controller" in solver_input:
                controller_type = solver_input["power_controller"].get("type", "").lower()
                if controller_type == "rl":
                    if "controller_params" not in solver_input["power_controller"]:
                        solver_input["power_controller"]["controller_params"] = {}
                    solver_input["power_controller"]["controller_params"]["rl_model_path"] = str(checkpoint_file.absolute())
                    print(f"Using checkpoint: {checkpoint_file.absolute()}")

    # -------------------------------
    # Launch GO-MELT Simulation
    # -------------------------------
    print("Running GO-MELT")
    print(f"GPU: {DEVICE_ID}, Input File: {input_file}")
    if checkpoints_dir:
        print(f"Checkpoints Directory: {checkpoints_dir}")
        print(f"Checkpoint Name: {checkpoint_name}")
    go_melt(solver_input, input_file)
