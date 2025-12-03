#!/usr/bin/env python3
"""
Create an animated GIF from GO-MELT VTK results, similar to the example visualization.

This script creates a 3D temperature field animation showing the evolution
of the thermal field during the simulation.

Usage:
    python create_animation.py [results_directory] [--level LEVEL] [--output OUTPUT_FILE]
    python create_animation.py [results_directory] --overview [--output OUTPUT_FILE]
    
Examples:
    python create_animation.py results/example/
    python create_animation.py results/example/ --level 3 --output my_animation.gif
    python create_animation.py results/example/ --overview --output overview.gif
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
    
    # Set output filename
    if output_file is None:
        output_file = f"animation_level{level}.gif"
    
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
            
            # Remove previous text if it exists
            if text_actor is not None:
                plotter.remove_actor(text_actor)
            
            # Add text with statistics (top-left corner)
            stats_text = (f"Level 3 Statistics:\n"
                         f"Max Temperature: {max_temp:.2f} K\n"
                         f"Avg Temperature: {avg_temp:.2f} K")
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
    
    # Set output filename
    if output_file is None:
        output_file = "animation_overview.gif"
    
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
    opacity_levels = {1: 0.3, 2: 0.6, 3: 1.0}
    
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
            
            # Remove previous text if it exists
            if text_actor is not None:
                plotter.remove_actor(text_actor)
            
            # Add text with statistics (top-left corner)
            stats_text = (f"Level 3 Statistics:\n"
                         f"Max Temperature: {max_temp:.2f} K\n"
                         f"Avg Temperature: {avg_temp:.2f} K")
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
        default='results/example/',
        help='Directory containing GO-MELT results (default: results/example/)'
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

