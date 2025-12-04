#!/usr/bin/env python3
"""
GO-MELT Simulation Runner with RL Power Control

This script runs the GO-MELT simulator with RL (Reinforcement Learning) control for power adjustment.
It uses the configuration from config/rl.json by default.
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
from controller.controller import RLController


def main():
    """Main function to run simulation with RL control."""
    # Clear terminal for clean output
    os.system("clear")
    
    # Parse Command-Line Arguments
    parser = argparse.ArgumentParser(
        description="GO-MELT Simulation Runner with RL Control",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python rl_control.py
  python rl_control.py config/rl.json
  python rl_control.py config/rl.json 0
        """
    )
    
    parser.add_argument(
        "input_file",
        type=str,
        nargs="?",
        default="config/rl.json",
        help="Path to input JSON configuration file (default: config/rl.json)"
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
    
    if controller_type != "rl":
        print(f"Warning: Controller type is '{controller_type}', not 'rl'. RL control may not work correctly.")
    
    # Get RL controller parameters
    target_temperature = power_controller_config.get("target_temperature", 3000.0)
    update_interval = power_controller_config.get("update_interval", 1)
    controller_params = power_controller_config.get("controller_params", {})
    rl_model_path = controller_params.get("rl_model_path", None)
    
    # Get base power from properties
    base_power = solver_input.get("properties", {}).get("laser_power", 285.0)
    
    # Launch GO-MELT Simulation
    print("Running GO-MELT with RL Power Control")
    print(f"GPU: {DEVICE_ID}, Input File: {input_file}")
    print(f"Target Temperature: {target_temperature} K")
    print(f"Base Power: {base_power} W")
    print(f"Update Interval: {update_interval} steps")
    if rl_model_path:
        print(f"RL Model Path: {rl_model_path}")
    else:
        print("RL Model Path: Not specified (will use linear controller fallback)")
    print(f"Power Range: {controller_params.get('power_min', 0.0)} - {controller_params.get('power_max', 500.0)} W")
    print("-" * 80)
    
    # Create simulator instance
    simulator = GoMeltSimulator(solver_input, input_file)
    
    # Create RL controller
    rl_controller = RLController(
        target_temperature=target_temperature,
        base_power=base_power,
        update_interval=update_interval,
        controller_params=controller_params,
        rl_model_path=rl_model_path
    )
    
    # Set flag to print observation breakdown on first update
    rl_controller._print_first_observation = True
    
    # Current power (starts at base power)
    current_power = base_power
    
    # Run simulation step by step with RL control
    while True:
        # Get power from RL controller
        new_power, power_updated = rl_controller.step(
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
    
    # Print RL controller statistics
    print("\n" + "=" * 80)
    print("RL Controller Statistics:")
    print("=" * 80)
    history = rl_controller.get_history()
    if history['power']:
        import numpy as np
        powers = np.array(history['power'])
        print(f"Total power updates: {len(history['power'])}")
        print(f"Mean power: {np.mean(powers):.2f} W")
        print(f"Std power: {np.std(powers):.2f} W")
        print(f"Max power: {np.max(powers):.2f} W")
        print(f"Min power: {np.min(powers):.2f} W")
        print(f"Final power: {current_power:.2f} W")
        if rl_controller.use_rl:
            print(f"Controller: RL Agent ({type(rl_controller.rl_model).__name__})")
        else:
            print("Controller: Linear (RL model not loaded)")
    print("=" * 80)


if __name__ == "__main__":
    main()

