#!/usr/bin/env python3
"""
Create an animated GIF from GO-MELT VTK results, similar to the example visualization.

This script creates a 3D temperature field animation showing the evolution
of the thermal field during the simulation.

Usage:
    python gif_creator.py [results_directory] [--level LEVEL] [--output OUTPUT_FILE]
    python gif_creator.py [results_directory] --overview [--output OUTPUT_FILE]
    
    If no results_directory is provided, the script will automatically find
    the latest results directory based on modification time.
    
Examples:
    python gif_creator.py                          # Auto-detect latest results
    python gif_creator.py results/example/         # Use specific directory
    python gif_creator.py --level 3 --output my_animation.gif  # Auto-detect with options
    python gif_creator.py results/example/ --overview --output overview.gif
"""

import argparse
import os
import glob
import sys

try:
    import pyvista as pv
    HAS_PYVISTA = True
except ImportError:
    print("Error: pyvista is required for 3D animation.")
    print("Install with: pip install pyvista")
    sys.exit(1)

try:
    import imageio
    HAS_IMAGEIO = True
except ImportError:
    print("Warning: imageio not found. Will use pyvista's built-in animation.")
    print("For better GIF quality, install with: pip install imageio imageio-ffmpeg")
    HAS_IMAGEIO = False

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    print("Error: numpy is required.")
    sys.exit(1)


def find_latest_results_dir(search_dirs=None):
    """
    Find the latest results directory based on modification time.
    
    Parameters:
    -----------
    search_dirs : list of str, optional
        Directories to search in for results. If None, searches 'results' and current directory.
        
    Returns:
    --------
    str or None
        Path to the latest results directory, or None if none found
    """
    if search_dirs is None:
        search_dirs = ['results', '.']
    
    # Find all directories that contain VTK files
    candidate_dirs = []
    
    for search_dir in search_dirs:
        if not os.path.exists(search_dir):
            continue
            
        # Check if search_dir itself contains VTK files
        vtk_files = glob.glob(os.path.join(search_dir, "Level*_*.vtr"))
        if vtk_files:
            candidate_dirs.append(os.path.abspath(search_dir))
        
        # Search in subdirectories of search_dir
        if os.path.isdir(search_dir):
            try:
                for item in os.listdir(search_dir):
                    item_path = os.path.join(search_dir, item)
                    if os.path.isdir(item_path):
                        # Check if this directory contains VTK files
                        vtk_files = glob.glob(os.path.join(item_path, "Level*_*.vtr"))
                        if vtk_files:
                            candidate_dirs.append(os.path.abspath(item_path))
            except PermissionError:
                # Skip directories we can't read
                continue
    
    if not candidate_dirs:
        return None
    
    # Find the directory with the latest modification time
    latest_dir = max(candidate_dirs, key=lambda d: os.path.getmtime(d))
    return latest_dir


def create_animation(results_dir, level=3, output_file=None, field='temperature', 
                     fps=10, quality=8, show_axes=True, show_colorbar=True):
    """
    Create an animated GIF from VTK files.
    
    Parameters:
    -----------
    results_dir : str
        Directory containing VTK files
    level : int
        Level number (1, 2, or 3)
    output_file : str
        Output filename (default: animation_level{level}.gif)
    field : str
        Field to visualize ('temperature' or 'state')
    fps : int
        Frames per second for animation
    quality : int
        GIF quality (1-10, higher is better quality but larger file)
    show_axes : bool
        Show coordinate axes
    show_colorbar : bool
        Show colorbar legend
    """
    # Find all VTK files for this level
    pattern = os.path.join(results_dir, f"Level{level}_*.vtr")
    files = sorted([f for f in glob.glob(pattern) if 'Final' not in f])
    
    if not files:
        print(f"Error: No Level {level} VTK files found in {results_dir}")
        return
    
    print(f"Found {len(files)} files for Level {level}")
    
    # Set output filename based on folder name
    if output_file is None:
        # Extract folder name from results directory
        folder_name = os.path.basename(os.path.normpath(results_dir))
        # Remove any trailing slashes and get just the folder name
        if not folder_name:
            folder_name = os.path.basename(os.path.dirname(results_dir))
        output_file = f"{folder_name}_level{level}.gif"
    
    # Ensure output directory exists (results/gifs)
    output_dir = "results/gifs"
    os.makedirs(output_dir, exist_ok=True)
    
    # If output_file is just a filename (no directory), save it to results/gifs
    # If it's already a full path (absolute or relative with directory), use it as is
    if not os.path.dirname(output_file):
        # Just a filename, save to results/gifs
        output_file = os.path.join(output_dir, output_file)
    elif not os.path.isabs(output_file):
        # Relative path with directory - still put in results/gifs for consistency
        output_file = os.path.join(output_dir, os.path.basename(output_file))
    # If absolute path, use it as is (no change needed)
    
    # Determine field name and colormap
    if field == 'temperature':
        field_name = 'Temperature (K)'
        cmap = 'hot'  # Hot colormap for temperature
        clim = None  # Auto-scale
    else:
        field_name = 'State (Powder/Solid)'
        cmap = 'viridis'
        clim = [0, 1]
    
    # Read first file to get bounds and setup
    print(f"Reading first file: {os.path.basename(files[0])}")
    grid = pv.read(files[0])
    
    # Get temperature range for consistent scaling
    if clim is None:
        all_temps = []
        for f in files[:min(10, len(files))]:  # Sample first 10 files
            g = pv.read(f)
            all_temps.extend(g[field_name])
        clim = [min(all_temps), max(all_temps)]
        print(f"Temperature range: {clim[0]:.2f} K to {clim[1]:.2f} K")
    
    # Create plotter
    plotter = pv.Plotter(off_screen=True)
    plotter.window_size = [1200, 800]
    plotter.set_background('white')  # White background like the example
    
    # Add initial mesh
    plotter.add_mesh(
        grid,
        scalars=field_name,
        cmap=cmap,
        clim=clim,
        show_edges=False,
        opacity=1.0,  # Fully opaque for better visibility
        scalar_bar_args={
            'title': field_name, 
            'vertical': True,
            'title_font_size': 14,
            'label_font_size': 12,
            'n_labels': 8
        } if show_colorbar else None
    )
    
    if show_axes:
        plotter.show_axes()
    
    # Set camera position (isometric view)
    plotter.camera_position = 'iso'
    plotter.camera.zoom(1.2)
    
    # Store initial camera position for consistency
    initial_camera = plotter.camera.position
    
    # Text actor reference for updating stats (will be created in loop)
    text_actor = None
    
    # Try to load MSE statistics if available
    mse_data = None
    mse_file = os.path.join(results_dir, "mse_stats.json")
    if os.path.exists(mse_file):
        try:
            import json
            with open(mse_file, "r") as f:
                mse_data = json.load(f)
            print(f"Loaded MSE statistics: MSE={mse_data.get('mse', 'N/A'):.2f} K², RMSE={mse_data.get('rmse', 'N/A'):.2f} K")
        except Exception as e:
            print(f"Warning: Could not load MSE statistics: {e}")
    
    # Create frames
    print("Creating animation frames...")
    frames = []
    
    for i, filepath in enumerate(files):
        if (i + 1) % 10 == 0:
            print(f"  Processing frame {i+1}/{len(files)}: {os.path.basename(filepath)}")
        
        # Read current file
        grid = pv.read(filepath)
        
        # Update mesh with new data
        plotter.update_scalars(grid[field_name], render=False)
        
        # Calculate and display Level 3 statistics if this is Level 3
        if level == 3 and field == 'temperature':
            temps = grid[field_name]
            max_temp = np.max(temps)
            avg_temp = np.mean(temps)
            
            # Try to read power from VTR file
            current_power = None
            if 'Laser Power (W)' in grid.point_data:
                power_data = grid['Laser Power (W)']
                # Power is constant field, get first value
                current_power = float(power_data[0]) if len(power_data) > 0 else None
            
            # Remove previous text if it exists
            if text_actor is not None:
                plotter.remove_actor(text_actor)
            
            # Add text with statistics (top-left corner)
            stats_text = (f"Level 3 Statistics:\n"
                         f"Max Temperature: {max_temp:.2f} K\n"
                         f"Avg Temperature: {avg_temp:.2f} K")
            if current_power is not None:
                stats_text += f"\nCurrent Power: {current_power:.2f} W"
            if mse_data is not None:
                stats_text += (f"\n\nControl Performance:\n"
                             f"Target: {mse_data.get('target_temperature', 0):.0f} K\n"
                             f"MSE: {mse_data.get('mse', 0):.2f} K²\n"
                             f"RMSE: {mse_data.get('rmse', 0):.2f} K")
            
            text_actor = plotter.add_text(
                stats_text,
                position='upper_left',
                font_size=12,
                color='black',
                shadow=True
            )
        
        # Keep camera position consistent
        plotter.camera.position = initial_camera
        
        # Render frame
        plotter.render()
        
        # Capture frame
        frame = plotter.screenshot(transparent_background=False)
        frames.append(frame)
    
    print(f"Rendering animation to {output_file}...")
    
    # Save as GIF
    if HAS_IMAGEIO:
        # Use imageio for better quality
        imageio.mimsave(
            output_file,
            frames,
            fps=fps,
            loop=0,  # Loop forever
            duration=1.0/fps
        )
    else:
        # Fallback: use pyvista's built-in method
        plotter.open_gif(output_file)
        for frame in frames:
            plotter.write_frame()
        plotter.close()
    
    print(f"Animation saved to {output_file}")
    print(f"  Frames: {len(frames)}")
    print(f"  FPS: {fps}")
    print(f"  Duration: {len(frames)/fps:.2f} seconds")


def create_overview_animation(results_dir, output_file=None, field='temperature', 
                              fps=10, show_axes=True, show_colorbar=True):
    """
    Create an overview animation showing all levels (1, 2, 3) combined in their
    actual spatial positions, showing the hierarchical nested structure.
    
    Parameters:
    -----------
    results_dir : str
        Directory containing VTK files
    output_file : str
        Output filename (default: animation_overview.gif)
    field : str
        Field to visualize ('temperature' or 'state')
    fps : int
        Frames per second for animation
    show_axes : bool
        Show coordinate axes
    show_colorbar : bool
        Show colorbar legend
    """
    # Find all VTK files for all levels
    all_files = {}
    for level in [1, 2, 3]:
        pattern = os.path.join(results_dir, f"Level{level}_*.vtr")
        files = sorted([f for f in glob.glob(pattern) if 'Final' not in f])
        if files:
            all_files[level] = files
    
    if not all_files:
        print(f"Error: No VTK files found in {results_dir}")
        return
    
    # Find the maximum number of files across all levels
    max_files = max(len(files) for files in all_files.values())
    print(f"Found files: Level 1: {len(all_files.get(1, []))}, "
          f"Level 2: {len(all_files.get(2, []))}, "
          f"Level 3: {len(all_files.get(3, []))}")
    
    # Set output filename based on folder name
    if output_file is None:
        # Extract folder name from results directory
        folder_name = os.path.basename(os.path.normpath(results_dir))
        # Remove any trailing slashes and get just the folder name
        if not folder_name:
            folder_name = os.path.basename(os.path.dirname(results_dir))
        output_file = f"{folder_name}_overview.gif"
    
    # Ensure output directory exists (results/gifs)
    output_dir = "results/gifs"
    os.makedirs(output_dir, exist_ok=True)
    
    # If output_file is just a filename (no directory), save it to results/gifs
    # If it's already a full path (absolute or relative with directory), use it as is
    if not os.path.dirname(output_file):
        # Just a filename, save to results/gifs
        output_file = os.path.join(output_dir, output_file)
    elif not os.path.isabs(output_file):
        # Relative path with directory - still put in results/gifs for consistency
        output_file = os.path.join(output_dir, os.path.basename(output_file))
    # If absolute path, use it as is (no change needed)
    
    # Determine field name and colormap
    if field == 'temperature':
        field_name = 'Temperature (K)'
        cmap = 'hot'
        clim = None
    else:
        field_name = 'State (Powder/Solid)'
        cmap = 'viridis'
        clim = [0, 1]
    
    # Get temperature range for consistent scaling across all levels
    if clim is None:
        all_temps = []
        for level, files in all_files.items():
            for f in files[:min(10, len(files))]:  # Sample first 10 files
                g = pv.read(f)
                if field_name in g.point_data:
                    all_temps.extend(g[field_name])
        if all_temps:
            clim = [min(all_temps), max(all_temps)]
            print(f"Temperature range: {clim[0]:.2f} K to {clim[1]:.2f} K")
        else:
            clim = None
    
    # Create single plotter (not subplots) to combine all levels
    plotter = pv.Plotter(off_screen=True)
    plotter.window_size = [1200, 800]
    plotter.set_background('white')
    
    # Store meshes and actors for each level
    meshes = {}
    actors = {}
    
    # Define opacity levels: Level 1 (coarsest) more transparent, Level 3 (finest) fully opaque
    opacity_levels = {1: 0.5, 2: 0.7, 3: 0.9}
    
    # Initialize all levels in the same plot
    print("Loading initial meshes for all levels...")
    for level in sorted(all_files.keys()):
        first_file = all_files[level][0]
        print(f"  Reading Level {level} first file: {os.path.basename(first_file)}")
        grid = pv.read(first_file)
        meshes[level] = grid
        
        # Add mesh with level-specific opacity
        actor = plotter.add_mesh(
            grid,
            scalars=field_name,
            cmap=cmap,
            clim=clim,
            show_edges=False,
            opacity=opacity_levels.get(level, 1.0),
            scalar_bar_args=None  # We'll add one colorbar for all
        )
        actors[level] = actor
    
    # Add single colorbar for all levels
    if show_colorbar:
        plotter.add_scalar_bar(
            title=field_name,
            vertical=True,
            title_font_size=14,
            label_font_size=12,
            n_labels=8
        )
    
    if show_axes:
        plotter.show_axes()
    
    # Set camera position (isometric view)
    plotter.camera_position = 'iso'
    plotter.camera.zoom(1.2)
    
    # Store initial camera position for consistency
    initial_camera = plotter.camera.position
    
    # Text actor reference for updating stats (will be created in loop)
    text_actor = None
    
    # Try to load MSE statistics if available
    mse_data = None
    mse_file = os.path.join(results_dir, "mse_stats.json")
    if os.path.exists(mse_file):
        try:
            import json
            with open(mse_file, "r") as f:
                mse_data = json.load(f)
            print(f"Loaded MSE statistics: MSE={mse_data.get('mse', 'N/A'):.2f} K², RMSE={mse_data.get('rmse', 'N/A'):.2f} K")
        except Exception as e:
            print(f"Warning: Could not load MSE statistics: {e}")
    
    # Create frames
    print("Creating combined overview animation frames...")
    frames = []
    
    for frame_idx in range(max_files):
        if (frame_idx + 1) % 10 == 0:
            print(f"  Processing frame {frame_idx+1}/{max_files}")
        
        # Store Level 3 grid for statistics
        level3_grid = None
        
        # Update each level
        for level in sorted(all_files.keys()):
            files = all_files[level]
            file_idx = min(frame_idx, len(files) - 1)
            filepath = files[file_idx]
            
            # Read current file
            grid = pv.read(filepath)
            
            # Store Level 3 grid for statistics
            if level == 3:
                level3_grid = grid
            
            # Update mesh with new data
            # Note: We remove and re-add to ensure proper updates when geometry might change
            plotter.remove_actor(actors[level])
            actors[level] = plotter.add_mesh(
                grid,
                scalars=field_name,
                cmap=cmap,
                clim=clim,
                show_edges=False,
                opacity=opacity_levels.get(level, 1.0),
                scalar_bar_args=None
            )
        
        # Calculate and display Level 3 statistics after all levels are updated
        if level3_grid is not None and field == 'temperature':
            temps = level3_grid[field_name]
            max_temp = np.max(temps)
            avg_temp = np.mean(temps)
            
            # Try to read power from Level 3 VTR file
            current_power = None
            if 'Laser Power (W)' in level3_grid.point_data:
                power_data = level3_grid['Laser Power (W)']
                # Power is constant field, get first value
                current_power = float(power_data[0]) if len(power_data) > 0 else None
            
            # Remove previous text if it exists
            if text_actor is not None:
                plotter.remove_actor(text_actor)
            
            # Add text with statistics (top-left corner)
            stats_text = (f"Level 3 Statistics:\n"
                         f"Max Temperature: {max_temp:.2f} K\n"
                         f"Avg Temperature: {avg_temp:.2f} K")
            if current_power is not None:
                stats_text += f"\nCurrent Power: {current_power:.2f} W"
            if mse_data is not None:
                stats_text += (f"\n\nControl Performance:\n"
                             f"Target: {mse_data.get('target_temperature', 0):.0f} K\n"
                             f"MSE: {mse_data.get('mse', 0):.2f} K²\n"
                             f"RMSE: {mse_data.get('rmse', 0):.2f} K")
            
            text_actor = plotter.add_text(
                stats_text,
                position='upper_left',
                font_size=12,
                color='black',
                shadow=True
            )
        
        # Keep camera position consistent
        plotter.camera.position = initial_camera
        
        # Render frame
        plotter.render()
        
        # Capture frame
        frame = plotter.screenshot(transparent_background=False)
        frames.append(frame)
    
    print(f"Rendering combined overview animation to {output_file}...")
    
    # Save as GIF
    if HAS_IMAGEIO:
        imageio.mimsave(
            output_file,
            frames,
            fps=fps,
            loop=0,
            duration=1.0/fps
        )
    else:
        plotter.open_gif(output_file)
        for frame in frames:
            plotter.write_frame()
        plotter.close()
    
    print(f"Combined overview animation saved to {output_file}")
    print(f"  Frames: {len(frames)}")
    print(f"  FPS: {fps}")
    print(f"  Duration: {len(frames)/fps:.2f} seconds")
    print(f"  Levels shown: {', '.join(f'Level {l}' for l in sorted(all_files.keys()))}")
    print(f"  Opacity: Level 1 (30%), Level 2 (60%), Level 3 (100%)")


def main():
    parser = argparse.ArgumentParser(
        description='Create animated GIF from GO-MELT VTK results',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        'results_dir',
        type=str,
        nargs='?',
        default=None,
        help='Directory containing GO-MELT results (default: auto-detect latest results directory)'
    )
    
    parser.add_argument(
        '--level',
        type=int,
        choices=[1, 2, 3],
        default=3,
        help='Level to visualize (default: 3)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output filename (default: animation_level{level}.gif)'
    )
    
    parser.add_argument(
        '--field',
        type=str,
        choices=['temperature', 'state'],
        default='temperature',
        help='Field to visualize (default: temperature)'
    )
    
    parser.add_argument(
        '--fps',
        type=int,
        default=10,
        help='Frames per second (default: 10)'
    )
    
    parser.add_argument(
        '--no-axes',
        action='store_true',
        help='Hide coordinate axes'
    )
    
    parser.add_argument(
        '--no-colorbar',
        action='store_true',
        help='Hide colorbar'
    )
    
    parser.add_argument(
        '--overview',
        action='store_true',
        help='Create overview animation showing all levels together (ignores --level)'
    )
    
    args = parser.parse_args()
    
    # Auto-detect latest results directory if not provided
    if args.results_dir is None:
        print("No results directory specified. Searching for latest results directory...")
        args.results_dir = find_latest_results_dir()
        if args.results_dir is None:
            print("Error: No results directory found. Please specify a directory or ensure")
            print("       results exist in a 'results/' subdirectory or current directory.")
            sys.exit(1)
        print(f"Using latest results directory: {args.results_dir}")
    
    # Check if results directory exists
    if not os.path.isdir(args.results_dir):
        print(f"Error: Results directory '{args.results_dir}' not found.")
        sys.exit(1)
    
    # Create animation
    if args.overview:
        create_overview_animation(
            args.results_dir,
            output_file=args.output,
            field=args.field,
            fps=args.fps,
            show_axes=not args.no_axes,
            show_colorbar=not args.no_colorbar
        )
    else:
        create_animation(
            args.results_dir,
            level=args.level,
            output_file=args.output,
            field=args.field,
            fps=args.fps,
            show_axes=not args.no_axes,
            show_colorbar=not args.no_colorbar
        )


if __name__ == '__main__':
    main()

