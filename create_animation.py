#!/usr/bin/env python3
"""
Create an animated GIF from GO-MELT VTK results, similar to the example visualization.

This script creates a 3D temperature field animation showing the evolution
of the thermal field during the simulation.

Usage:
    python create_animation.py [results_directory] [--level LEVEL] [--output OUTPUT_FILE]
    
Examples:
    python create_animation.py results/example/
    python create_animation.py results/example/ --level 3 --output my_animation.gif
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
    
    args = parser.parse_args()
    
    # Check if results directory exists
    if not os.path.isdir(args.results_dir):
        print(f"Error: Results directory '{args.results_dir}' not found.")
        sys.exit(1)
    
    # Create animation
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

