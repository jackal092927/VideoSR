#!/usr/bin/env python3
"""
Generate videos from CSV trajectory data.
Creates visualization videos showing ideal vs noisy trajectories,
with options to add physics equations and trajectory analysis.
"""

import numpy as np
import pandas as pd
import cv2
from pathlib import Path
import argparse
import matplotlib.pyplot as plt
from matplotlib import animation
import matplotlib.patches as patches
import re


def parse_equations(equations_text):
    """
    Parse discovered physics equations from text format.
    
    Args:
        equations_text: String containing equations like "114.960 1 + -0.028 x0 + 0.078 x1"
    
    Returns:
        List of parsed equation coefficients
    """
    equations = []
    
    for line in equations_text.strip().split('\n'):
        if not line.strip():
            continue
            
        # Parse equation coefficients
        # Format: "coefficient term1 + coefficient term2 + ..."
        parts = line.split('+')
        coeffs = {}
        
        for part in parts:
            part = part.strip()
            if not part:
                continue
                
            # Split coefficient and term
            if ' ' in part:
                coeff_str, term = part.split(' ', 1)
                try:
                    coeff = float(coeff_str)
                    coeffs[term.strip()] = coeff
                except ValueError:
                    continue
            else:
                # Just a coefficient (constant term)
                try:
                    coeff = float(part)
                    coeffs['1'] = coeff
                except ValueError:
                    continue
        
        if coeffs:
            equations.append(coeffs)
    
    return equations


def simulate_trajectory_from_equations(equations, t_values, initial_conditions=None):
    """
    Simulate trajectory using discovered physics equations.
    
    Args:
        equations: List of parsed equation coefficients
        t_values: Time array
        initial_conditions: Dict with initial x, y, vx, vy values
    
    Returns:
        DataFrame with simulated trajectory
    """
    
    if len(equations) < 2:
        print("Warning: Need at least 2 equations for x and y simulation")
        return None
    
    # Default initial conditions
    if initial_conditions is None:
        initial_conditions = {
            'x': 74.0,  # From your real data
            'y': 390.0, # From your real data
            'vx': 170.0, # From your real data
            'vy': -158.0 # From your real data
        }
    
    # Initialize arrays
    n_frames = len(t_values)
    x_sim = np.zeros(n_frames)
    y_sim = np.zeros(n_frames)
    vx_sim = np.zeros(n_frames)
    vy_sim = np.zeros(n_frames)
    
    # Set initial conditions
    x_sim[0] = initial_conditions['x']
    y_sim[0] = initial_conditions['y']
    vx_sim[0] = initial_conditions['vx']
    vy_sim[0] = initial_conditions['vy']
    
    # Simulate using the discovered equations
    dt = t_values[1] - t_values[0] if len(t_values) > 1 else 0.033
    
    for i in range(1, n_frames):
        # Get current state
        x_curr = x_sim[i-1]
        y_curr = y_sim[i-1]
        vx_curr = vx_sim[i-1]
        vy_curr = vy_sim[i-1]
        
        # Apply equation 1 (likely for x acceleration)
        if len(equations) > 0:
            eq1 = equations[0]
            ax = 0.0
            if '1' in eq1:
                ax += eq1['1']
            if 'x0' in eq1:
                ax += eq1['x0'] * x_curr
            if 'x1' in eq1:
                ax += eq1['x1'] * y_curr
            if 'x0^2' in eq1:
                ax += eq1['x0^2'] * x_curr**2
            if 'x1^2' in eq1:
                ax += eq1['x1^2'] * y_curr**2
            if 'x0 x1' in eq1:
                ax += eq1['x0 x1'] * x_curr * y_curr
        
        # Apply equation 2 (likely for y acceleration)
        if len(equations) > 1:
            eq2 = equations[1]
            ay = 0.0
            if '1' in eq2:
                ay += eq2['1']
            if 'x0' in eq2:
                ay += eq2['x0'] * x_curr
            if 'x1' in eq2:
                ay += eq2['x1'] * y_curr
            if 'x0^2' in eq2:
                ay += eq2['x0^2'] * x_curr**2
            if 'x1^2' in eq2:
                ay += eq2['x1^2'] * y_curr**2
            if 'x0 x1' in eq2:
                ay += eq2['x0 x1'] * x_curr * y_curr
        
        # Update velocities and positions using Euler integration
        vx_sim[i] = vx_curr + ax * dt
        vy_sim[i] = vy_curr + ay * dt
        x_sim[i] = x_curr + vx_curr * dt
        y_sim[i] = y_curr + vy_curr * dt
    
    # Create DataFrame
    data = {
        'frame': np.arange(n_frames),
        't': t_values,
        'x': x_sim,
        'y': y_sim,
        'vx': vx_sim,
        'vy': vy_sim,
        'ax': np.gradient(vx_sim, t_values),
        'ay': np.gradient(vy_sim, t_values),
        'area': np.ones(n_frames) * 100,  # Default area
        'ok': np.ones(n_frames)  # All frames are valid
    }
    
    return pd.DataFrame(data)


def create_reconstructed_trajectory_video(equations_text, output_path, width=800, height=600, fps=30.0, 
                                       initial_conditions=None, duration=4.0):
    """
    Create a video showing trajectory reconstructed from discovered physics equations.
    
    Args:
        equations_text: Text containing discovered equations
        output_path: Output video file path
        width: Video width
        height: Video height
        fps: Frames per second
        initial_conditions: Initial position and velocity
        duration: Video duration in seconds
    """
    
    print("Creating reconstructed trajectory video from discovered equations...")
    
    # Parse equations
    equations = parse_equations(equations_text)
    if not equations:
        print("Error: Could not parse equations")
        return False
    
    print(f"Parsed {len(equations)} equations:")
    for i, eq in enumerate(equations):
        print(f"  Equation {i+1}: {eq}")
    
    # Generate time array
    n_frames = int(duration * fps)
    t_values = np.linspace(0, duration, n_frames)
    
    # Simulate trajectory
    df_sim = simulate_trajectory_from_equations(equations, t_values, initial_conditions)
    if df_sim is None:
        return False
    
    # Create video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    colors = {
        'background': (240, 240, 240),
        'trajectory': (0, 100, 255),    # Orange
        'current_pos': (0, 255, 0),     # Green
        'text': (0, 0, 0),
        'grid': (200, 200, 200),
        'equations': (128, 0, 128)      # Purple
    }
    
    print(f"Generating {n_frames} frames...")
    
    for frame_idx in range(n_frames):
        # Create background
        frame = np.full((height, width, 3), colors['background'], dtype=np.uint8)
        
        # Add grid
        grid_spacing = 50
        for x in range(0, width, grid_spacing):
            cv2.line(frame, (x, 0), (x, height), colors['grid'], 1)
        for y in range(0, height, grid_spacing):
            cv2.line(frame, (0, y), (width, y), colors['grid'], 1)
        
        # Draw trajectory up to current frame
        if frame_idx > 1:
            for i in range(1, min(frame_idx + 1, len(df_sim))):
                pt1 = (int(df_sim.iloc[i-1]['x']), int(df_sim.iloc[i-1]['y']))
                pt2 = (int(df_sim.iloc[i]['x']), int(df_sim.iloc[i]['y']))
                
                if (0 <= pt1[0] < width and 0 <= pt1[1] < height and 
                    0 <= pt2[0] < width and 0 <= pt2[1] < height):
                    cv2.line(frame, pt1, pt2, colors['trajectory'], 2)
        
        # Draw current position
        if frame_idx < len(df_sim):
            row = df_sim.iloc[frame_idx]
            x, y = int(row['x']), int(row['y'])
            
            if 0 <= x < width and 0 <= y < height:
                cv2.circle(frame, (x, y), 8, colors['current_pos'], -1)
                cv2.circle(frame, (x, y), 8, colors['text'], 2)
                
                # Draw position info
                info_text = f"t={row['t']:.3f}s  pos=({x},{y})"
                cv2.putText(frame, info_text, (10, height-60), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.7, colors['text'], 2)
        
        # Draw equations
        cv2.putText(frame, "RECONSTRUCTED TRAJECTORY", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, colors['text'], 2)
        cv2.putText(frame, "From Discovered Physics Equations:", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, colors['text'], 2)
        
        # Draw frame info
        frame_text = f"Frame: {frame_idx+1}/{n_frames}"
        cv2.putText(frame, frame_text, (10, height-30), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.7, colors['text'], 2)
        
        # Draw equations box
        box_width = 400
        box_height = 30 + len(equations) * 25
        box_x = width - box_width - 20
        box_y = 20
        
        # Semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (box_x, box_y), (box_x + box_width, box_y + box_height), 
                     (255, 255, 255), -1)
        cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)
        
        cv2.putText(frame, "Discovered Equations:", (box_x + 10, box_y + 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, colors['text'], 2)
        
        for i, eq in enumerate(equations[:3]):
            y_pos = box_y + 45 + i * 25
            if y_pos < height - 20:
                eq_str = f"{i+1}. {eq}"
                cv2.putText(frame, eq_str, (box_x + 10, y_pos), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors['text'], 1)
        
        out.write(frame)
        
        if frame_idx % 10 == 0:
            print(f"  Frame {frame_idx+1}/{n_frames}")
    
    out.release()
    print(f"Reconstructed trajectory video saved to: {output_path}")
    return True


def create_comparison_video_real_vs_reconstructed(real_csv, equations_text, output_path, 
                                                width=800, height=600, fps=30.0):
    """
    Create a side-by-side comparison video of real vs reconstructed trajectories.
    """
    
    print("Creating comparison video: Real vs Reconstructed trajectories...")
    
    # Load real trajectory data
    df_real = pd.read_csv(real_csv)
    
    # Parse equations and simulate reconstructed trajectory
    equations = parse_equations(equations_text)
    if not equations:
        print("Error: Could not parse equations")
        return False
    
    # Use same time values as real data
    t_values = df_real['t'].values
    df_sim = simulate_trajectory_from_equations(equations, t_values)
    if df_sim is None:
        return False
    
    # Create side-by-side video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width*2, height))
    
    colors = {
        'real': (0, 0, 255),      # Red
        'reconstructed': (0, 255, 0),  # Green
        'background': (240, 240, 240),
        'text': (0, 0, 0),
        'grid': (200, 200, 200)
    }
    
    min_frames = min(len(df_real), len(df_sim))
    
    for frame_idx in range(min_frames):
        # Create combined frame
        frame = np.full((height, width*2, 3), colors['background'], dtype=np.uint8)
        
        # Left side: Real trajectory
        left_frame = frame[:, :width]
        left_frame = create_background(left_frame, width, height, colors)
        left_frame = draw_trajectory(left_frame, df_real, frame_idx, colors['real'])
        left_frame = draw_current_info(left_frame, df_real, frame_idx, colors['real'])
        cv2.putText(left_frame, "REAL TRAJECTORY", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, colors['text'], 2)
        
        # Right side: Reconstructed trajectory
        right_frame = frame[:, width:]
        right_frame = create_background(right_frame, width, height, colors)
        right_frame = draw_trajectory(right_frame, df_sim, frame_idx, colors['reconstructed'])
        right_frame = draw_current_info(right_frame, df_sim, frame_idx, colors['reconstructed'])
        cv2.putText(right_frame, "RECONSTRUCTED", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, colors['text'], 2)
        
        # Combine frames
        frame[:, :width] = left_frame
        frame[:, width:] = right_frame
        
        # Add comparison info
        if frame_idx < len(df_real):
            real_x, real_y = int(df_real.iloc[frame_idx]['x']), int(df_real.iloc[frame_idx]['y'])
            sim_x, sim_y = int(df_sim.iloc[frame_idx]['x']), int(df_sim.iloc[frame_idx]['y'])
            
            # Calculate error
            error = np.sqrt((real_x - sim_x)**2 + (real_y - sim_y)**2)
            cv2.putText(frame, f"Error: {error:.1f} pixels", (width//2-100, height-20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, colors['text'], 2)
        
        out.write(frame)
        
        if frame_idx % 10 == 0:
            print(f"  Frame {frame_idx+1}/{min_frames}")
    
    out.release()
    print(f"Comparison video saved to: {output_path}")
    return True


def create_trajectory_video(csv_path, output_path, video_type="trajectory", 
                           width=800, height=600, fps=30.0, 
                           show_equations=None, show_velocity=False):
    """
    Create a video from CSV trajectory data.
    
    Args:
        csv_path: Path to CSV file with trajectory data
        output_path: Output video file path
        video_type: "trajectory", "comparison", or "analysis"
        width: Video width in pixels
        height: Video height in pixels
        fps: Frames per second
        show_equations: List of equations to display
        show_velocity: Whether to show velocity vectors
    """
    
    # Load trajectory data
    print(f"Loading trajectory data from: {csv_path}")
    df = pd.read_csv(csv_path)
    
    # Ensure required columns exist
    required_cols = ['t', 'x', 'y']
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"CSV must contain columns: {required_cols}")
    
    # Video writer setup
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Color definitions
    colors = {
        'background': (240, 240, 240),  # Light gray
        'trajectory': (0, 100, 255),    # Orange
        'current_pos': (0, 255, 0),     # Green
        'text': (0, 0, 0),              # Black
        'grid': (200, 200, 200),        # Light gray
        'velocity': (255, 0, 0),        # Red
        'acceleration': (128, 0, 128)   # Purple
    }
    
    # Create background
    def create_background():
        frame = np.full((height, width, 3), colors['background'], dtype=np.uint8)
        
        # Add grid
        grid_spacing = 50
        for x in range(0, width, grid_spacing):
            cv2.line(frame, (x, 0), (x, height), colors['grid'], 1)
        for y in range(0, height, grid_spacing):
            cv2.line(frame, (0, y), (width, y), colors['grid'], 1)
        
        # Add coordinate labels
        for x in range(0, width, grid_spacing):
            cv2.putText(frame, str(x), (x+5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors['text'], 1)
        for y in range(0, height, grid_spacing):
            cv2.putText(frame, str(y), (5, y+15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors['text'], 1)
        
        return frame
    
    # Draw trajectory path
    def draw_trajectory(frame, df, current_frame_idx):
        if current_frame_idx < 2:
            return frame
        
        # Draw trajectory line
        for i in range(1, min(current_frame_idx + 1, len(df))):
            if i < len(df) and i-1 < len(df):
                pt1 = (int(df.iloc[i-1]['x']), int(df.iloc[i-1]['y']))
                pt2 = (int(df.iloc[i]['x']), int(df.iloc[i]['y']))
                
                # Check if points are valid
                if (0 <= pt1[0] < width and 0 <= pt1[1] < height and 
                    0 <= pt2[0] < width and 0 <= pt2[1] < height):
                    cv2.line(frame, pt1, pt2, colors['trajectory'], 2)
        
        return frame
    
    # Draw current position and info
    def draw_current_info(frame, df, current_frame_idx):
        if current_frame_idx >= len(df):
            return frame
        
        row = df.iloc[current_frame_idx]
        x, y = int(row['x']), int(row['y'])
        
        # Check if position is valid
        if 0 <= x < width and 0 <= y < height:
            # Draw current position
            cv2.circle(frame, (x, y), 8, colors['current_pos'], -1)
            cv2.circle(frame, (x, y), 8, colors['text'], 2)
            
            # Draw velocity vector if requested
            if show_velocity and 'vx' in df.columns and 'vy' in df.columns:
                vx, vy = row['vx'], row['vy']
                if np.isfinite(vx) and np.isfinite(vy):
                    # Scale velocity for visualization
                    scale = 0.1
                    end_x = int(x + vx * scale)
                    end_y = int(y + vy * scale)
                    if (0 <= end_x < width and 0 <= end_y < height):
                        cv2.arrowedLine(frame, (x, y), (end_x, end_y), colors['velocity'], 3)
            
            # Draw position info
            info_text = f"t={row['t']:.3f}s  pos=({x},{y})"
            cv2.putText(frame, info_text, (10, height-60), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.7, colors['text'], 2)
            
            # Draw frame info
            frame_text = f"Frame: {current_frame_idx+1}/{len(df)}"
            cv2.putText(frame, frame_text, (10, height-30), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.7, colors['text'], 2)
        
        return frame
    
    # Draw physics equations
    def draw_equations(frame, equations):
        if not equations:
            return frame
        
        # Draw equations box
        box_width = 400
        box_height = 30 + len(equations) * 25
        box_x = width - box_width - 20
        box_y = 20
        
        # Semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (box_x, box_y), (box_x + box_width, box_y + box_height), 
                     (255, 255, 255), -1)
        cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)
        
        # Draw equations
        cv2.putText(frame, "Discovered Equations:", (box_x + 10, box_y + 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, colors['text'], 2)
        
        for i, eq in enumerate(equations[:3]):  # Show first 3 equations
            y_pos = box_y + 45 + i * 25
            if y_pos < height - 20:
                cv2.putText(frame, f"{i+1}. {eq}", (box_x + 10, y_pos), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors['text'], 1)
        
        return frame
    
    # Generate video frames
    print(f"Generating {len(df)} frames...")
    
    for frame_idx in range(len(df)):
        # Create background
        frame = create_background()
        
        # Draw trajectory
        frame = draw_trajectory(frame, df, frame_idx)
        
        # Draw current position and info
        frame = draw_current_info(frame, df, frame_idx)
        
        # Draw equations if provided
        if show_equations:
            frame = draw_equations(frame, show_equations)
        
        # Write frame
        out.write(frame)
        
        # Progress indicator
        if frame_idx % 10 == 0:
            print(f"  Frame {frame_idx+1}/{len(df)}")
    
    out.release()
    print(f"Video saved to: {output_path}")


def create_comparison_video(noisy_csv, ideal_csv, output_path, width=800, height=600, fps=30.0):
    """
    Create a side-by-side comparison video of noisy vs ideal trajectories.
    """
    
    print("Creating comparison video...")
    
    # Load both datasets
    df_noisy = pd.read_csv(noisy_csv)
    df_ideal = pd.read_csv(ideal_csv)
    
    # Ensure same length
    min_frames = min(len(df_noisy), len(df_ideal))
    df_noisy = df_noisy.iloc[:min_frames]
    df_ideal = df_ideal.iloc[:min_frames]
    
    # Create side-by-side video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width*2, height))
    
    colors = {
        'noisy': (0, 0, 255),      # Red
        'ideal': (0, 255, 0),      # Green
        'background': (240, 240, 240),
        'text': (0, 0, 0),
        'grid': (200, 200, 200)
    }
    
    for frame_idx in range(min_frames):
        # Create combined frame
        frame = np.full((height, width*2, 3), colors['background'], dtype=np.uint8)
        
        # Left side: Noisy trajectory
        left_frame = frame[:, :width]
        left_frame = create_background(left_frame, width, height, colors)
        left_frame = draw_trajectory(left_frame, df_noisy, frame_idx, colors['noisy'])
        left_frame = draw_current_info(left_frame, df_noisy, frame_idx, colors['noisy'])
        cv2.putText(left_frame, "NOISY TRAJECTORY", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, colors['text'], 2)
        
        # Right side: Ideal trajectory
        right_frame = frame[:, width:]
        right_frame = create_background(right_frame, width, height, colors)
        right_frame = draw_trajectory(right_frame, df_ideal, frame_idx, colors['ideal'])
        right_frame = draw_current_info(right_frame, df_ideal, frame_idx, colors['ideal'])
        cv2.putText(right_frame, "IDEAL TRAJECTORY", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, colors['text'], 2)
        
        # Combine frames
        frame[:, :width] = left_frame
        frame[:, width:] = right_frame
        
        # Add comparison info
        if 'ok' in df_noisy.columns:
            noisy_ok = df_noisy.iloc[frame_idx]['ok'] if frame_idx < len(df_noisy) else 1
            status = "GOOD" if noisy_ok == 1 else "OUTLIER"
            cv2.putText(frame, f"Frame Status: {status}", (width//2-100, height-20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, colors['text'], 2)
        
        out.write(frame)
        
        if frame_idx % 10 == 0:
            print(f"  Frame {frame_idx+1}/{min_frames}")
    
    out.release()
    print(f"Comparison video saved to: {output_path}")


def create_background(frame, width, height, colors):
    """Helper function to create background with grid."""
    # Add grid
    grid_spacing = 50
    for x in range(0, width, grid_spacing):
        cv2.line(frame, (x, 0), (x, height), colors['grid'], 1)
    for y in range(0, height, grid_spacing):
        cv2.line(frame, (0, y), (width, y), colors['grid'], 1)
    return frame


def draw_trajectory(frame, df, current_frame_idx, color):
    """Helper function to draw trajectory."""
    if current_frame_idx < 2:
        return frame
    
    for i in range(1, min(current_frame_idx + 1, len(df))):
        if i < len(df) and i-1 < len(df):
            pt1 = (int(df.iloc[i-1]['x']), int(df.iloc[i-1]['y']))
            pt2 = (int(df.iloc[i]['x']), int(df.iloc[i]['y']))
            
            if (0 <= pt1[0] < frame.shape[1] and 0 <= pt1[1] < frame.shape[0] and 
                0 <= pt2[0] < frame.shape[1] and 0 <= pt2[1] < frame.shape[0]):
                cv2.line(frame, pt1, pt2, color, 2)
    
    return frame


def draw_current_info(frame, df, current_frame_idx, color):
    """Helper function to draw current position info."""
    if current_frame_idx >= len(df):
        return frame
    
    row = df.iloc[current_frame_idx]
    x, y = int(row['x']), int(row['y'])
    
    if 0 <= x < frame.shape[1] and 0 <= y < frame.shape[0]:
        cv2.circle(frame, (x, y), 8, color, -1)
        cv2.circle(frame, (x, y), 8, (0, 0, 0), 2)
    
    return frame


def main():
    """Main function with command line interface."""
    
    parser = argparse.ArgumentParser(description="Generate videos from CSV trajectory data")
    parser.add_argument("--csv", help="Path to CSV file with trajectory data")
    parser.add_argument("--output", required=True, help="Output video file path")
    parser.add_argument("--type", choices=["trajectory", "comparison", "reconstructed", "real_vs_reconstructed"], 
                       default="trajectory", help="Type of video to generate")
    parser.add_argument("--ideal-csv", help="Path to ideal trajectory CSV for comparison")
    parser.add_argument("--width", type=int, default=800, help="Video width")
    parser.add_argument("--height", type=int, default=600, help="Video height")
    parser.add_argument("--fps", type=float, default=30.0, help="Frames per second")
    parser.add_argument("--equations", help="Path to equations text file")
    parser.add_argument("--show-velocity", action="store_true", help="Show velocity vectors")
    parser.add_argument("--duration", type=float, default=4.0, help="Duration for reconstructed video (seconds)")
    
    args = parser.parse_args()
    
    # Create output directory
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    
    if args.type == "reconstructed":
        # Generate reconstructed trajectory from equations
        if not args.equations:
            print("Error: --equations required for reconstructed video type")
            return
        
        with open(args.equations, 'r') as f:
            equations_text = f.read()
        
        create_reconstructed_trajectory_video(
            equations_text, args.output, args.width, args.height, args.fps, 
            duration=args.duration
        )
        
    elif args.type == "real_vs_reconstructed":
        # Generate comparison video: real vs reconstructed
        if not args.csv or not args.equations:
            print("Error: --csv and --equations required for real_vs_reconstructed video type")
            return
        
        with open(args.equations, 'r') as f:
            equations_text = f.read()
        
        create_comparison_video_real_vs_reconstructed(
            args.csv, equations_text, args.output, args.width, args.height, args.fps
        )
        
    elif args.type == "comparison" and args.ideal_csv:
        create_comparison_video(args.csv, args.ideal_csv, args.output, 
                              args.width, args.height, args.fps)
    else:
        # Regular trajectory video
        if not args.csv:
            print("Error: --csv required for trajectory video type")
            return
            
        # Load equations if provided
        equations = None
        if args.equations and Path(args.equations).exists():
            with open(args.equations, 'r') as f:
                equations = [line.strip() for line in f.readlines() if line.strip()]
        
        create_trajectory_video(args.csv, args.output, "trajectory",
                              args.width, args.height, args.fps,
                              equations, args.show_velocity)


if __name__ == "__main__":
    main()
