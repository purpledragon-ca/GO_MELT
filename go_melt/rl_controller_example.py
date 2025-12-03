"""
Example usage of the RL Controller for GO-MELT power control.

This demonstrates how to use the linear controller functions in a simulation context.
"""

from rl_controller import (
    get_max_temperature_level3,
    linear_power_controller,
    compute_power_action,
    PowerController
)
import numpy as np


def example_usage_functions():
    """Example of using individual controller functions."""
    print("=== Example: Using Individual Functions ===\n")
    
    # Simulate Levels dictionary (in real usage, this comes from GO-MELT simulation)
    # Create a dummy temperature field for Level 3
    dummy_temps = np.array([
        [2800, 2850, 2900],
        [3000, 3100, 3200],  # Max temp is 3200K
        [2700, 2750, 2800]
    ])
    
    Levels = {
        3: {
            "T0": dummy_temps
        }
    }
    
    # 1. Get observation (max temperature)
    max_temp = get_max_temperature_level3(Levels)
    print(f"Current max temperature: {max_temp:.2f} K")
    
    # 2. Compute power adjustment using linear controller
    target_temp = 3000.0  # Target max temperature
    current_power = 285.0  # Current power in Watts
    kp = 0.1  # Proportional gain (W/K)
    
    new_power = linear_power_controller(
        max_temperature=max_temp,
        target_temperature=target_temp,
        current_power=current_power,
        kp=kp,
        power_min=0.0,
        power_max=500.0
    )
    
    print(f"Current power: {current_power:.2f} W")
    print(f"Temperature error: {target_temp - max_temp:.2f} K")
    print(f"New power: {new_power:.2f} W")
    print(f"Power change: {new_power - current_power:.2f} W\n")


def example_usage_compute_action():
    """Example of using the compute_power_action interface."""
    print("=== Example: Using compute_power_action Interface ===\n")
    
    # Observation (max temperature)
    observation = 3200.0  # Current max temp
    
    # Compute action
    new_power = compute_power_action(
        observation=observation,
        target_temperature=3000.0,
        current_power=285.0,
        controller_params={
            'kp': 0.1,
            'power_min': 0.0,
            'power_max': 500.0
        }
    )
    
    print(f"Observation (max temp): {observation:.2f} K")
    print(f"Target temperature: 3000.0 K")
    print(f"Action (new power): {new_power:.2f} W\n")


def example_usage_controller_class():
    """Example of using the PowerController class."""
    print("=== Example: Using PowerController Class ===\n")
    
    # Initialize controller
    controller = PowerController(
        target_temperature=3000.0,
        base_power=285.0,
        update_interval=10,  # Update every 10 steps (0.1m)
        controller_params={
            'kp': 0.1,
            'power_min': 0.0,
            'power_max': 500.0
        }
    )
    
    # Simulate a few steps
    print("Simulating 25 steps (should update at steps 10, 20):\n")
    
    for step in range(1, 26):
        # Create dummy Levels with varying temperatures
        # Temperature increases over time
        temp_value = 2800 + step * 15  # Simulate temperature rising
        dummy_temps = np.array([[temp_value]])
        
        Levels = {
            3: {
                "T0": dummy_temps
            }
        }
        
        # Execute control step
        new_power, updated = controller.step(Levels)
        
        if updated:
            max_temp = get_max_temperature_level3(Levels)
            print(f"Step {step:2d}: Power updated to {new_power:.2f} W "
                  f"(max temp: {max_temp:.2f} K)")
        elif step == 1:
            print(f"Step {step:2d}: Using initial power {new_power:.2f} W")
    
    print("\nControl History:")
    history = controller.get_history()
    print(f"  Observations: {len(history['observations'])} updates")
    print(f"  Final observation: {history['observations'][-1]:.2f} K" if history['observations'] else "  No observations yet")
    print(f"  Final power: {history['actions'][-1]:.2f} W" if history['actions'] else "  No actions yet")


if __name__ == "__main__":
    print("RL Controller Usage Examples\n")
    print("=" * 50 + "\n")
    
    example_usage_functions()
    print("=" * 50 + "\n")
    
    example_usage_compute_action()
    print("=" * 50 + "\n")
    
    example_usage_controller_class()
    print("\n" + "=" * 50)
    print("\nThese functions are ready to be integrated into go_melt.py")
    print("and will later be replaced by a trained RL agent.")

