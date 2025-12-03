import os
import sys
import time
import argparse
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

    level_names = ["L1", "L2", "L3"]

    # -------------------------------
    # Setup: Properties, Mesh, Nonmesh
    # -------------------------------
    Properties = SetupProperties(solver_input.get("properties", {}))
    Levels = SetupLevels(solver_input, Properties)
    Nonmesh = SetupNonmesh(solver_input.get("nonmesh", {}))

    # -------------------------------
    # Static Mesh Metadata
    # -------------------------------
    ne_nn = getStaticNodesAndElements(Levels)
    subcycle = getStaticSubcycle(Nonmesh)

    # -------------------------------
    # Mesh Ratios for Movement Logic
    # -------------------------------
    L1L2Eratio = [
        int(jnp.round(Levels[1]["h"][i] / Levels[2]["h"][i])) for i in range(2)
    ] + [int(jnp.round(Properties["layer_height"] / Levels[2]["h"][2]))]

    L2L3Eratio = [
        int(jnp.round(Levels[2]["h"][i] / Levels[3]["h"][i])) for i in range(3)
    ]

    # -------------------------------
    # Initialize Power Controller (PID or RL) - before other initialization
    # -------------------------------
    power_controller = None
    controller_type = None
    
    if HAS_CONTROLLERS:
        # Get controller configuration from solver_input
        controller_config = solver_input.get("power_controller", {})
        controller_type = controller_config.get("type", "none").lower()  # "pid", "rl", or "none"
        
        if controller_type in ["pid", "rl"]:
            # Common parameters
            target_temp = controller_config.get("target_temperature", 3000.0)
            base_power = Properties["laser_power"]
            update_interval = controller_config.get("update_interval", 10)  # Every 10 steps (0.1m)
            controller_params = controller_config.get("controller_params", {})
            
            try:
                if controller_type == "pid":
                    # Initialize PID controller
                    if PIDController is not None:
                        # Set default PID parameters if not provided
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
                    # Initialize RL controller
                    if RLController is not None:
                        # Set default RL parameters if not provided
                        if "kp" not in controller_params:
                            controller_params["kp"] = 0.1
                        if "power_min" not in controller_params:
                            controller_params["power_min"] = 0.0
                        if "power_max" not in controller_params:
                            controller_params["power_max"] = 500.0
                        
                        # Check for RL model path
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
                        
                        # Flag to print observation on first use
                        power_controller._print_first_observation = True
                    else:
                        print("Warning: RLController (RL) not available. Power control disabled.")
                
            except Exception as e:
                print(f"Warning: Failed to initialize {controller_type.upper()} controller: {e}")
                power_controller = None
        else:
            # No controller or invalid type
            if controller_type != "none":
                print(f"Warning: Unknown controller type '{controller_type}'. Valid options: 'pid', 'rl', 'none'")
            print("Running without power controller (using toolpath power).")

    # -------------------------------
    # Update Save Path with Timestamp and Controller Type/JSON Name
    # -------------------------------
    # Generate timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Extract JSON name from input file (without extension)
    if input_file:
        json_name = Path(input_file).stem  # Get filename without extension
    else:
        # Fallback: use controller type or "none"
        json_name = controller_type if controller_type else "none"
    
    # Get base save path from Nonmesh
    base_save_path = Nonmesh.get("save_path", "./results/")
    
    # Create new save path: base_path/timestamp_jsonname/
    # If save_path is "./results/example/", we want "./results/timestamp_jsonname/"
    base_path = Path(base_save_path.rstrip("/")).parent
    new_save_path = base_path / f"{timestamp}_{json_name}"
    
    # Ensure directory exists
    new_save_path.mkdir(parents=True, exist_ok=True)
    new_save_path_str = str(new_save_path) + "/"
    
    # Update Nonmesh save_path
    Nonmesh["save_path"] = new_save_path_str
    
    # Update toolpath path to be in the new save directory
    original_toolpath = Nonmesh.get("toolpath", "./results/example/toolpath.txt")
    toolpath_filename = Path(original_toolpath).name
    new_toolpath = new_save_path / toolpath_filename
    
    # If using existing txt file, copy it to new location
    if Nonmesh.get("use_txt", 0):
        original_toolpath_path = Path(original_toolpath)
        if original_toolpath_path.exists():
            import shutil
            shutil.copy2(original_toolpath_path, new_toolpath)
            print(f"Copied toolpath file to: {new_toolpath}")
        else:
            print(f"Warning: Toolpath file not found at {original_toolpath}")
    
    Nonmesh["toolpath"] = str(new_toolpath)
    
    print(f"Results will be saved to: {new_save_path_str}")

    # -------------------------------
    # Toolpath Parsing
    # -------------------------------
    if Nonmesh["use_txt"]:
        move_mesh = count_lines(Nonmesh["toolpath"])
    else:
        move_mesh = parsingGcode(Nonmesh, Properties, Levels[2]["h"])

    total_t_inc = move_mesh  # Total time steps

    # -------------------------------
    # Initial Laser Position
    # -------------------------------
    if not Properties["laser_center"]:
        with open(Nonmesh["toolpath"], "r") as tool_path_file:
            laser_start = np.array(
                [float(val) for val in tool_path_file.readline().split(",")]
            )
    else:
        laser_start = np.array(Properties["laser_center"])

    # -------------------------------
    # Interpolation Matrices
    # -------------------------------
    L1L2Interp = interpolatePointsMatrix(Levels[1], Levels[2]["node_coords"])
    L2L3Interp = interpolatePointsMatrix(Levels[2], Levels[3]["node_coords"])
    LInterp = [L1L2Interp, L2L3Interp]

    # -------------------------------
    # Time & Output Initialization
    # -------------------------------
    time_inc = record_inc = wait_inc = 0
    t_output = 0.0
    savenum = int(time_inc / Nonmesh["record_step"]) + 1
    # Get initial power for saving
    initial_power = Properties["laser_power"]
    if power_controller is not None:
        initial_power = power_controller.get_current_power()
    saveResults(Levels, Nonmesh, savenum, power=initial_power)
    
    # -------------------------------
    # Error Tracking for MSE Calculation
    # -------------------------------
    temperature_errors = []  # Store squared errors: (target_temp - max_temp)^2
    target_temperature = None
    if power_controller is not None:
        target_temperature = power_controller.target_temperature

    # -------------------------------
    # Layer Tracking & Accumulation
    # -------------------------------
    laser_prev_z = float("inf")
    _dwell_time_count = 0
    record_accum = True

    if record_accum:
        accum_time = jnp.zeros(Levels[0]["nn"])
        max_accum_time = jnp.zeros(Levels[0]["nn"])

    move_hist = [jnp.array(0), jnp.array(0), jnp.array(0)]  # Laser movement history

    # -------------------------------
    # Simulation Flags
    # -------------------------------
    force_move = move_vert = new_checkpoint = False
    ongoing_simulation = single_step = True

    # -------------------------------
    # Toolpath File & Checkpointing
    # -------------------------------
    tool_path_file = open(Nonmesh["toolpath"], "r")
    np_path = Path(Nonmesh["save_path"] + "checkpoint").absolute()
    layer_check = Nonmesh["layer_num"] + Nonmesh["restart_layer_num"]

    # -------------------------------
    # Load Checkpoint if Requested
    # -------------------------------
    if Nonmesh["layer_num"] > 0:
        # Loading checkpoint for Layer {Nonmesh['layer_num']}
        print(f"Loading checkpoint: Layer {Nonmesh['layer_num']}")
        FILENAME = f"Checkpoint{str(Nonmesh['layer_num']).zfill(4)}.pkl"

        with open(np_path.joinpath(FILENAME), "rb") as f:
            Levels, accum_time, max_accum_time, time_inc_loaded, record_inc = dill.load(
                f
            )

        load_chkpt = True
        line_len = len(tool_path_file.readline())
        tool_path_file.seek(time_inc_loaded * line_len)
        time_inc += time_inc_loaded
    else:
        load_chkpt = False

    # -------------------------------
    # Warm-up moveEverything to avoid GPU graph capture errors
    # -------------------------------
    # Force JAX to compile moveEverything before the main loop
    # This prevents "Failed to capture gpu graph" errors, especially with PID controller
    if not load_chkpt:
        try:
            print("Warming up JAX compilation...")
            # Ensure moveEverything is available (imported via computeFunctions import *)
            # Import explicitly to ensure it's in scope
            from computeFunctions import moveEverything
            
            # Create dummy inputs for warm-up (use same position to avoid actual movement)
            # Use explicit float32 dtype for consistency
            warmup_v = jnp.array([laser_start[0], laser_start[1], laser_start[2], 0.0, 0.0, 0.0, Properties["laser_power"]], dtype=jnp.float32)
            warmup_move_hist = [jnp.array(0, dtype=jnp.float32), jnp.array(0, dtype=jnp.float32), jnp.array(0, dtype=jnp.float32)]
            # Warm-up call - this forces JAX to compile without GPU graph capture
            # Use block_until_ready() to ensure compilation happens
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
            # Force execution to complete - handle tuple result properly
            if isinstance(result, tuple):
                _ = [x.block_until_ready() if hasattr(x, 'block_until_ready') else None for x in result if x is not None]
            else:
                _ = result.block_until_ready() if hasattr(result, 'block_until_ready') else None
            # Clear cache after warm-up to ensure fresh compilation in main loop
            if hasattr(moveEverything, '_clear_cache'):
                moveEverything._clear_cache()
            print("Warm-up complete.")
        except Exception as e:
            print(f"Warning: Warm-up failed: {e}")
            print("This may cause GPU graph capture errors. Continuing anyway...")
            import traceback
            traceback.print_exc()

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
                    trying_flag = True
                    tmp_coords = copy.deepcopy(Levels[1]["orig_node_coords"])
                    _L1T_state_idx = 0

                    # Find matching z-layer in Level 1
                    while trying_flag:
                        if jnp.isclose(
                            tmp_coords[2] - laser_pos[2], 0, atol=1e-4
                        ).any():
                            trying_flag = False
                        else:
                            tmp_coords[2] += Properties["layer_height"]
                            _L1T_state_idx += 1

                    # Update Level 1 state if not loading from checkpoint
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

                    # Update interpolation matrices and static node/element info
                    L1L2Interp = interpolatePointsMatrix(
                        Levels[1], Levels[2]["node_coords"]
                    )
                    L2L3Interp = interpolatePointsMatrix(
                        Levels[2], Levels[3]["node_coords"]
                    )
                    LInterp = [L1L2Interp, L2L3Interp]
                    tmp_ne_nn = calcStaticTmpNodesAndElements(Levels, laser_pos)

                    # Update layer tracking and flags
                    laser_prev_z = laser_pos[2]
                    force_move = True
                    wait_inc = 0
                    move_vert = True

                    if not load_chkpt:
                        # Save Level 0 state at the start of a new layer
                        saveState(
                            Levels[0],
                            "Level0_",
                            Nonmesh["layer_num"],
                            Nonmesh["save_path"],
                            0,
                        )

                        if record_accum:
                            # Save accumulated melt time
                            accum_time = jnp.maximum(accum_time, max_accum_time)
                            jnp.savez(
                                Nonmesh["save_path"]
                                + "accum_time"
                                + str(Nonmesh["layer_num"]).zfill(4),
                                accum_time=accum_time,
                            )

                        # Shift Level 0 data down to simulate vertical mesh movement
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

                        Levels[0]["S1"] = (
                            Levels[0]["S1"].at[:_0nn2].set(Levels[0]["S1"][_0nn1:])
                        )
                        Levels[0]["S1"] = Levels[0]["S1"].at[_0nn2:].set(0)

                        # Update z-coordinates for Level 0
                        Levels[0]["node_coords"][2] = (
                            Levels[0]["orig_node_coords"][2]
                            + laser_pos[2]
                            - Levels[0]["orig_node_coords"][2][-1]
                        )

                        if record_accum:
                            max_accum_time = jnp.zeros(Levels[0]["nn"])
                            accum_time = accum_time.at[:_0nn2].set(accum_time[_0nn1:])
                            accum_time = accum_time.at[_0nn2:].set(0)

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

    # Save final Level 0 state and temperature fields
    saveState(Levels[0], "Level0_", Nonmesh["layer_num"], Nonmesh["save_path"], 0)
    # Get final power for saving
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

    if record_accum:
        accum_time = jnp.maximum(accum_time, max_accum_time)
        jnp.savez(
            Nonmesh["save_path"] + "accum_time" + str(Nonmesh["layer_num"]).zfill(4),
            accum_time=accum_time,
        )

    # -------------------------------
    # Calculate and Print Mean Squared Error (MSE)
    # -------------------------------
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
        
        # Save MSE to JSON file for animation script
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

    # Clear JAX caches
    try:
        stepGOMELT._clear_cache()
        stepGOMELTDwellTime._clear_cache()
        subcycleGOMELT._clear_cache()
        moveEverything._clear_cache()
        gc.collect()
        # Cache cleared
    except:
        gc.collect()

    print("\nSimulation completed.")


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
