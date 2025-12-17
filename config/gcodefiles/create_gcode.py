#!/usr/bin/env python3
"""
Generate G-code file based on example.json configuration.
Takes input x (number of steps), y (number of parallel paths), z (number of layers)
and creates G-code with delta_h = 0.04 (layer_height).
"""

import json
import sys
import os
import argparse
from pathlib import Path


def create_gcode(x_steps, y_paths, z_layers, config_path, output_path, delta_h=0.04):
    """
    Create G-code file based on configuration and dimensions.
    
    Args:
        x_steps: Number of steps in X direction
        y_paths: Number of parallel paths in Y direction
        z_layers: Number of layers in Z direction
        config_path: Path to example.json configuration file
        output_path: Path to output G-code file
        delta_h: Layer height (default 0.04)
        Note: y_gap is read from laser_radius in the configuration file
    """
    try:
        # Read configuration file
        with open(config_path, 'r') as f:
            config = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in configuration file: {e}")
    except Exception as e:
        raise RuntimeError(f"Error reading configuration file: {e}")
    
    try:
        # Get laser center and laser_radius from properties
        properties = config.get("properties")
        if properties is None:
            raise ValueError("Configuration file missing 'properties' section")
        
        laser_center = properties.get("laser_center")
        if laser_center is None:
            raise ValueError("Configuration file missing 'laser_center' in properties")
        
        if len(laser_center) < 3:
            raise ValueError("laser_center must have at least 3 elements [x, y, z]")
        
        start_x_base = float(laser_center[0])
        start_y_base = float(laser_center[1])
        start_z_base = float(laser_center[2])
        
        # Get laser_radius for Y gap between parallel paths
        laser_radius = properties.get("laser_radius")
        if laser_radius is None:
            raise ValueError("Configuration file missing 'laser_radius' in properties")
        
        y_gap = float(laser_radius)
        if y_gap <= 0:
            raise ValueError("laser_radius must be positive")
        
    except (KeyError, IndexError, ValueError, TypeError) as e:
        raise ValueError(f"Error extracting properties from configuration: {e}")
    
    try:
        # Get timestep_L3 and laser_velocity from nonmesh section
        nonmesh = config.get("nonmesh")
        if nonmesh is None:
            raise ValueError("Configuration file missing 'nonmesh' section")
        
        timestep_L3 = nonmesh.get("timestep_L3")
        if timestep_L3 is None:
            raise ValueError("Configuration file missing 'timestep_L3' in nonmesh")
        
        laser_velocity = nonmesh.get("laser_velocity")
        if laser_velocity is None:
            raise ValueError("Configuration file missing 'laser_velocity' in nonmesh")
        
        timestep_L3 = float(timestep_L3)
        laser_velocity = float(laser_velocity)
        
        if timestep_L3 <= 0:
            raise ValueError("timestep_L3 must be positive")
        if laser_velocity <= 0:
            raise ValueError("laser_velocity must be positive")
        
        # Validate x_steps
        if x_steps < 1:
            raise ValueError("x_steps must be at least 1")
        
        # Calculate X distance based on timestep_L3, laser_velocity, and x_steps
        # distance = velocity * time_per_step * number_of_steps
        x_distance = laser_velocity * timestep_L3 * x_steps
        
        # Use laser_center X as start position
        x_start = start_x_base
        # End position is start + calculated distance
        x_end = x_start + x_distance
        
    except (KeyError, IndexError, ValueError, TypeError) as e:
        raise ValueError(f"Error extracting parameters from configuration: {e}")
    
    try:
        # Validate inputs
        if y_paths < 1:
            raise ValueError("y_paths must be at least 1")
        if z_layers < 1:
            raise ValueError("z_layers must be at least 1")
        if delta_h <= 0:
            raise ValueError("delta_h must be positive")
        
        # Calculate Y positions for parallel paths
        # Center the paths around the base Y position
        total_y_span = (y_paths - 1) * y_gap
        y_start_offset = -total_y_span / 2.0
        
        y_positions = []
        for i in range(y_paths):
            y_pos = start_y_base + y_start_offset + i * y_gap
            y_positions.append(y_pos)
        
    except (ValueError, ZeroDivisionError) as e:
        raise ValueError(f"Error calculating path dimensions: {e}")
    
    try:
        # Create output directory if it doesn't exist
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Generate G-code
        with open(output_path, 'w') as f:
            for layer in range(z_layers):
                z = start_z_base + (layer + 1) * delta_h
                
                # For each parallel path
                for path_idx, y_pos in enumerate(y_positions):
                    # Alternate direction for each path (back and forth)
                    reverse = (path_idx % 2 == 1)
                    
                    if reverse:
                        # Start from end, go to start
                        path_x_start = x_end
                        path_x_end = x_start
                    else:
                        # Start from start, go to end
                        path_x_start = x_start
                        path_x_end = x_end
                    
                    # Rapid move to start position
                    f.write(f"G0 X{path_x_start:.2f} Y{y_pos:.3f} Z{z:.3f}\n")
                    
                    # Linear move to end position (no intermediate steps)
                    f.write(f"G1 X{path_x_end:.2f} Y{y_pos:.3f} Z{z:.3f}\n")
                    
                    # Add blank line between paths (except after last path in layer)
                    if path_idx < len(y_positions) - 1:
                        f.write("\n")
                
                # Add blank line between layers (except after last layer)
                if layer < z_layers - 1:
                    f.write("\n")
        
        print(f"G-code file created successfully: {output_path}")
        print(f"X steps: {x_steps}, Y paths: {y_paths}, Z layers: {z_layers}")
        print(f"X range: {x_start:.2f} to {x_end:.2f} (distance: {x_distance:.6f})")
        print(f"Calculated from: laser_velocity={laser_velocity}, timestep_L3={timestep_L3}")
        print(f"Y positions: {[f'{y:.3f}' for y in y_positions]}")
        print(f"Layer height: {delta_h}, Y gap (laser_radius): {y_gap}")
        
    except IOError as e:
        raise IOError(f"Error writing G-code file: {e}")
    except Exception as e:
        raise RuntimeError(f"Error generating G-code: {e}")


def main():
    """Main function to parse arguments and generate G-code."""
    parser = argparse.ArgumentParser(
        description="Generate G-code file based on example.json configuration",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "x",
        type=int,
        help="Number of steps in X direction"
    )
    
    parser.add_argument(
        "y",
        type=int,
        help="Number of parallel paths in Y direction"
    )
    
    parser.add_argument(
        "z",
        type=int,
        help="Number of layers in Z direction"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        help="Path to example.json configuration file"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        help="Path to output G-code file"
    )
    
    parser.add_argument(
        "--delta_h",
        type=float,
        help="Layer height (default: 0.04)"
    )
    
    args = parser.parse_args()
    
    # Get workspace root (parent of config directory)
    script_dir = Path(__file__).parent
    workspace_root = script_dir.parent.parent
    
    # Set default config path if not provided
    try:
        if args.config is None:
            config_path = workspace_root / "config" / "example.json"
        else:
            config_path = Path(args.config)
            if not config_path.is_absolute():
                config_path = workspace_root / config_path
    except Exception as e:
        print(f"Error: Invalid config path: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Set default output path if not provided
    try:
        if args.output is None:
            output_path = script_dir / "generated.gcode"
        else:
            output_path = Path(args.output)
            if not output_path.is_absolute():
                output_path = script_dir / output_path
    except Exception as e:
        print(f"Error: Invalid output path: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Set delta_h
    try:
        if args.delta_h is None:
            delta_h = 0.04
        else:
            delta_h = float(args.delta_h)
            if delta_h <= 0:
                raise ValueError("delta_h must be positive")
    except (ValueError, TypeError) as e:
        print(f"Error: Invalid delta_h: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Validate inputs
    try:
        if args.x < 1:
            raise ValueError("X steps must be at least 1")
        if args.y < 1:
            raise ValueError("Y paths must be at least 1")
        if args.z < 1:
            raise ValueError("Z layers must be at least 1")
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Generate G-code
    try:
        create_gcode(
            args.x,
            args.y,
            args.z,
            str(config_path),
            str(output_path),
            delta_h
        )
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

