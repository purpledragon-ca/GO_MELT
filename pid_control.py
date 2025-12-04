#!/usr/bin/env python3
"""
GO-MELT Simulation Runner with PID Power Control

This script runs the GO-MELT simulator with PID control for power adjustment.
It uses the configuration from config/pid.json by default.
"""

import os
import sys
import json
import argparse
import jax

base_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, base_dir)
sys.path.insert(0, os.path.join(base_dir, 'simulator'))

from simulator import GoMeltSimulator
from controller.controller import PIDController


def main():
    """Main function to run simulation with PID control."""
    # Clear terminal for clean output
    os.system("clear")
    
    # Parse Command-Line Arguments
    parser = argparse.ArgumentParser(
        description="GO-MELT Simulation Runner with PID Control",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python play.py
  python play.py config/pid.json
  python play.py config/pid.json 0
        """
    )
    
    parser.add_argument(
        "input_file",
        type=str,
        nargs="?",
        default="config/pid.json",
        help="Path to input JSON configuration file (default: config/pid.json)"
    )
    
    parser.add_argument(
        "device_id",
        type=int,
        nargs="?",
        default=0,
        help="GPU device ID (default: 0)"
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
    
    # Extract power controller configuration
    power_controller_config = solver_input.get("power_controller", {})
    controller_type = power_controller_config.get("type", "none")
    
    if controller_type != "pid":
        print(f"Warning: Controller type is '{controller_type}', not 'pid'. PID control may not work correctly.")
    
    # Get PID controller parameters
    target_temperature = power_controller_config.get("target_temperature", 3000.0)
    update_interval = power_controller_config.get("update_interval", 1)
    controller_params = power_controller_config.get("controller_params", {})
    
    # Get base power from properties
    base_power = solver_input.get("properties", {}).get("laser_power", 285.0)
    
    # Launch GO-MELT Simulation
    print("Running GO-MELT with PID Power Control")
    print(f"GPU: {DEVICE_ID}, Input File: {input_file}")
    print(f"Target Temperature: {target_temperature} K")
    print(f"Base Power: {base_power} W")
    print(f"Update Interval: {update_interval} steps")
    print(f"PID Gains: kp={controller_params.get('kp', 0.1)}, "
          f"ki={controller_params.get('ki', 0.01)}, "
          f"kd={controller_params.get('kd', 0.0)}")
    print("-" * 80)
    
    # Create simulator instance
    simulator = GoMeltSimulator(solver_input, input_file)
    
    # Create PID controller
    pid_controller = PIDController(
        target_temperature=target_temperature,
        base_power=base_power,
        update_interval=update_interval,
        controller_params=controller_params
    )
    
    # Current power (starts at base power)
    current_power = base_power
    
    # Run simulation step by step with PID control
    while True:
        # Get power from PID controller
        new_power, power_updated = pid_controller.step(
            simulator.Levels,
            Properties=simulator.Properties
        )
        
        # Update current power if controller updated it
        if power_updated:
            current_power = new_power
        
        # Run simulation step with current power
        if not simulator.nextstep(current_power):
            break
    
    # Finalize simulation
    simulator.finalize()
    
    # Print PID controller statistics
    print("\n" + "=" * 80)
    print("PID Controller Statistics:")
    print("=" * 80)
    history = pid_controller.get_history()
    if history['errors']:
        import numpy as np
        errors = np.array(history['errors'])
        absolute_errors = np.abs(errors)
        print(f"Total power updates: {len(history['power'])}")
        print(f"Mean error: {np.mean(absolute_errors):.2f} K")
        print(f"Std error: {np.std(absolute_errors):.2f} K")
        print(f"Max error: {np.max(absolute_errors):.2f} K")
        print(f"Min error: {np.min(absolute_errors):.2f} K")
        print(f"Final power: {current_power:.2f} W")
    print("=" * 80)


if __name__ == "__main__":
    main()

