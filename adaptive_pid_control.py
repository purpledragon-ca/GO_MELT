#!/usr/bin/env python3
"""
GO-MELT Simulation Runner with Adaptive PID Power Control

This script runs the GO-MELT simulator with adaptive PID control for power adjustment.
The adaptive PID controller dynamically adjusts its gains (Kp, Ki, Kd) based on:
- Error magnitude (large errors -> higher gains, small errors -> lower gains)
- Oscillation detection (reduces gains when oscillations are detected)
- Error rate of change (adjusts derivative gain for damping)

It uses the configuration from config/adaptive_pid.json by default.
"""

import os
import sys
import json
import argparse
import jax
import numpy as np

base_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, base_dir)
sys.path.insert(0, os.path.join(base_dir, 'simulator'))

from simulator import GoMeltSimulator
from controller.controller import AdaptivePIDController


def main():
    """Main function to run simulation with adaptive PID control."""
    # Clear terminal for clean output
    os.system("clear")
    
    # Parse Command-Line Arguments
    parser = argparse.ArgumentParser(
        description="GO-MELT Simulation Runner with Adaptive PID Control",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python adaptive_pid_control.py
  python adaptive_pid_control.py config/adaptive_pid.json
  python adaptive_pid_control.py config/adaptive_pid.json 0
        """
    )
    
    parser.add_argument(
        "input_file",
        type=str,
        nargs="?",
        default="config/adaptive_pid.json",
        help="Path to input JSON configuration file (default: config/adaptive_pid.json)"
    )
    
    parser.add_argument(
        "device_id",
        type=int,
        nargs="?",
        default=0,
        help="GPU device ID (default: 0)"
    )
    
    parser.add_argument(
        "--show-gains",
        action="store_true",
        help="Print gain values when they are updated (default: False)"
    )
    
    args = parser.parse_args()
    
    DEVICE_ID = args.device_id
    input_file = args.input_file
    show_gains = args.show_gains
    
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
    
    # Extract power controller configuration
    power_controller_config = solver_input.get("power_controller", {})
    controller_type = power_controller_config.get("type", "adaptive_pid")
    
    # Force adaptive PID if not already set
    if controller_type != "adaptive_pid":
        print(f"Warning: Controller type is '{controller_type}'. Forcing adaptive_pid mode.")
        controller_type = "adaptive_pid"
        power_controller_config["type"] = "adaptive_pid"
    
    # Get PID controller parameters
    target_temperature = power_controller_config.get("target_temperature", 3000.0)
    update_interval = power_controller_config.get("update_interval", 1)
    controller_params = power_controller_config.get("controller_params", {})
    
    # Get base power from properties
    base_power = solver_input.get("properties", {}).get("laser_power", 285.0)
    
    # Launch GO-MELT Simulation
    print("Running GO-MELT with Adaptive PID Power Control")
    print(f"GPU: {DEVICE_ID}, Input File: {input_file}")
    print(f"Target Temperature: {target_temperature} K")
    print(f"Base Power: {base_power} W")
    print(f"Update Interval: {update_interval} steps")
    print(f"\nBase PID Gains:")
    print(f"  Kp: {controller_params.get('kp', 0.1)}")
    print(f"  Ki: {controller_params.get('ki', 0.01)}")
    print(f"  Kd: {controller_params.get('kd', 0.0)}")
    print(f"\nAdaptation Parameters:")
    print(f"  Adaptation Rate: {controller_params.get('adaptation_rate', 0.1)}")
    print(f"  Adaptation Interval: {controller_params.get('adaptation_interval', 3)} updates")
    print(f"  Oscillation Window: {controller_params.get('oscillation_window_size', 5)} samples")
    print(f"  Error Thresholds: Large={controller_params.get('error_threshold_large', 100.0)} K, "
          f"Small={controller_params.get('error_threshold_small', 10.0)} K")
    print(f"  Oscillation Threshold: {controller_params.get('oscillation_threshold', 0.3)}")
    print(f"\nGain Bounds:")
    print(f"  Kp: [{controller_params.get('kp_min', 0.01)}, {controller_params.get('kp_max', 1.0)}]")
    print(f"  Ki: [{controller_params.get('ki_min', 0.001)}, {controller_params.get('ki_max', 0.1)}]")
    print(f"  Kd: [{controller_params.get('kd_min', 0.0)}, {controller_params.get('kd_max', 0.5)}]")
    print("-" * 80)
    
    # Create simulator instance
    simulator = GoMeltSimulator(solver_input, input_file)
    
    # Create adaptive PID controller
    adaptive_pid_controller = AdaptivePIDController(
        target_temperature=target_temperature,
        base_power=base_power,
        update_interval=update_interval,
        controller_params=controller_params
    )
    
    # Current power (starts at base power)
    current_power = base_power
    last_gains = adaptive_pid_controller.get_current_gains()
    
    # Run simulation step by step with adaptive PID control
    step_count = 0
    while True:
        # Get power from adaptive PID controller
        new_power, power_updated = adaptive_pid_controller.step(
            simulator.Levels,
            Properties=simulator.Properties
        )
        
        # Update current power if controller updated it
        if power_updated:
            current_power = new_power
            step_count += 1
            
            # Show gain updates if requested
            if show_gains:
                current_gains = adaptive_pid_controller.get_current_gains()
                if current_gains != last_gains:
                    print(f"Step {step_count * update_interval}: Gains updated - "
                          f"Kp={current_gains['kp']:.4f}, "
                          f"Ki={current_gains['ki']:.4f}, "
                          f"Kd={current_gains['kd']:.4f}")
                    last_gains = current_gains
        
        # Run simulation step with current power
        if not simulator.nextstep(current_power):
            break
    
    # Finalize simulation
    simulator.finalize()
    
    # Print adaptive PID controller statistics
    print("\n" + "=" * 80)
    print("Adaptive PID Controller Statistics:")
    print("=" * 80)
    history = adaptive_pid_controller.get_history()
    
    if history['errors']:
        errors = np.array(history['errors'])
        absolute_errors = np.abs(errors)
        
        print(f"\nPerformance Metrics:")
        print(f"  Total power updates: {len(history['power'])}")
        print(f"  Mean absolute error: {np.mean(absolute_errors):.2f} K")
        print(f"  Std deviation of error: {np.std(absolute_errors):.2f} K")
        print(f"  Max absolute error: {np.max(absolute_errors):.2f} K")
        print(f"  Min absolute error: {np.min(absolute_errors):.2f} K")
        print(f"  Final power: {current_power:.2f} W")
        
        # Print adaptive gains information
        if 'gains' in history and history['gains']:
            final_gains = history['gains'][-1]
            print(f"\nFinal Adaptive Gains:")
            print(f"  Kp: {final_gains['kp']:.4f} (base: {controller_params.get('kp', 0.1)}, "
                  f"change: {((final_gains['kp'] / controller_params.get('kp', 0.1)) - 1.0) * 100:+.1f}%)")
            print(f"  Ki: {final_gains['ki']:.4f} (base: {controller_params.get('ki', 0.01)}, "
                  f"change: {((final_gains['ki'] / controller_params.get('ki', 0.01)) - 1.0) * 100:+.1f}%)")
            print(f"  Kd: {final_gains['kd']:.4f} (base: {controller_params.get('kd', 0.0)}, "
                  f"change: {((final_gains['kd'] / max(controller_params.get('kd', 0.0), 0.001)) - 1.0) * 100:+.1f}%)" 
                  if controller_params.get('kd', 0.0) > 0 else f"change: N/A)")
            
            # Show gain evolution statistics
            if len(history['gains']) > 1:
                kp_values = np.array([g['kp'] for g in history['gains']])
                ki_values = np.array([g['ki'] for g in history['gains']])
                kd_values = np.array([g['kd'] for g in history['gains']])
                
                print(f"\nGain Evolution Statistics:")
                print(f"  Kp:")
                print(f"    Range: [{np.min(kp_values):.4f}, {np.max(kp_values):.4f}]")
                print(f"    Mean: {np.mean(kp_values):.4f}, Std: {np.std(kp_values):.4f}")
                print(f"    Initial: {kp_values[0]:.4f}, Final: {kp_values[-1]:.4f}")
                
                print(f"  Ki:")
                print(f"    Range: [{np.min(ki_values):.4f}, {np.max(ki_values):.4f}]")
                print(f"    Mean: {np.mean(ki_values):.4f}, Std: {np.std(ki_values):.4f}")
                print(f"    Initial: {ki_values[0]:.4f}, Final: {ki_values[-1]:.4f}")
                
                print(f"  Kd:")
                print(f"    Range: [{np.min(kd_values):.4f}, {np.max(kd_values):.4f}]")
                print(f"    Mean: {np.mean(kd_values):.4f}, Std: {np.std(kd_values):.4f}")
                print(f"    Initial: {kd_values[0]:.4f}, Final: {kd_values[-1]:.4f}")
                
                # Calculate gain adaptation frequency
                kp_changes = np.sum(np.diff(kp_values) != 0)
                ki_changes = np.sum(np.diff(ki_values) != 0)
                kd_changes = np.sum(np.diff(kd_values) != 0)
                
                print(f"\nGain Adaptation Activity:")
                print(f"  Kp adapted {kp_changes} times ({kp_changes/len(kp_values)*100:.1f}% of updates)")
                print(f"  Ki adapted {ki_changes} times ({ki_changes/len(ki_values)*100:.1f}% of updates)")
                print(f"  Kd adapted {kd_changes} times ({kd_changes/len(kd_values)*100:.1f}% of updates)")
    
    print("=" * 80)


if __name__ == "__main__":
    main()

