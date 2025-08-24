#!/usr/bin/env python3
"""
Compare tracking results with ground truth data
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def compare_tracking_results(output_csv, groundtruth_csv, output_dir="comparison_plots"):
    """Compare tracking results with ground truth."""
    
    # Load data
    tracked = pd.read_csv(output_csv)
    ground_truth = pd.read_csv(groundtruth_csv)
    
    # Create output directory
    Path(output_dir).mkdir(exist_ok=True)
    
    print(f"Loaded tracking data: {len(tracked)} points")
    print(f"Loaded ground truth data: {len(ground_truth)} points")
    
    # Align data by time (in case of slight differences)
    min_len = min(len(tracked), len(ground_truth))
    tracked = tracked.iloc[:min_len]
    ground_truth = ground_truth.iloc[:min_len]
    
    # Calculate errors
    pos_error_x = np.abs(tracked['x'] - ground_truth['x'])
    pos_error_y = np.abs(tracked['y'] - ground_truth['y'])
    
    # Calculate velocity errors if available
    if 'vx' in tracked.columns and 'vx' in ground_truth.columns:
        vel_error_x = np.abs(tracked['vx'] - ground_truth['vx'])
        vel_error_y = np.abs(tracked['vy'] - ground_truth['vy'])
    else:
        vel_error_x = vel_error_y = None
    
    # Calculate acceleration errors if available
    if 'ax' in tracked.columns and 'ax' in ground_truth.columns:
        acc_error_x = np.abs(tracked['ax'] - ground_truth['ax'])
        acc_error_y = np.abs(tracked['ay'] - ground_truth['ay'])
    else:
        acc_error_x = acc_error_y = None
    
    # Print statistics
    print(f"\n=== TRACKING ACCURACY ANALYSIS ===")
    print(f"Position Error (X): Mean = {pos_error_x.mean():.3f}, Max = {pos_error_x.max():.3f}, RMS = {np.sqrt(np.mean(pos_error_x**2)):.3f}")
    print(f"Position Error (Y): Mean = {pos_error_y.mean():.3f}, Max = {pos_error_y.max():.3f}, RMS = {np.sqrt(np.mean(pos_error_y**2)):.3f}")
    
    if vel_error_x is not None:
        print(f"Velocity Error (X): Mean = {vel_error_x.mean():.3f}, Max = {vel_error_x.max():.3f}, RMS = {np.sqrt(np.mean(vel_error_x**2)):.3f}")
        print(f"Velocity Error (Y): Mean = {vel_error_y.mean():.3f}, Max = {vel_error_y.max():.3f}, RMS = {np.sqrt(np.mean(vel_error_y**2)):.3f}")
    
    if acc_error_x is not None:
        print(f"Acceleration Error (X): Mean = {acc_error_x.mean():.3f}, Max = {acc_error_x.max():.3f}, RMS = {np.sqrt(np.mean(acc_error_x**2)):.3f}")
        print(f"Acceleration Error (Y): Mean = {acc_error_y.mean():.3f}, Max = {acc_error_y.max():.3f}, RMS = {np.sqrt(np.mean(acc_error_y**2)):.3f}")
    
    # Create comparison plots
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Tracking Results vs Ground Truth Comparison', fontsize=16)
    
    # Position plots
    axes[0, 0].plot(tracked['t'], tracked['x'], 'b-', label='Tracked', alpha=0.7)
    axes[0, 0].plot(ground_truth['t'], ground_truth['x'], 'r--', label='Ground Truth', alpha=0.7)
    axes[0, 0].set_xlabel('Time (s)')
    axes[0, 0].set_ylabel('X Position (pixels)')
    axes[0, 0].set_title('X Position')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[1, 0].plot(tracked['t'], tracked['y'], 'b-', label='Tracked', alpha=0.7)
    axes[1, 0].plot(ground_truth['t'], ground_truth['y'], 'r--', label='Ground Truth', alpha=0.7)
    axes[1, 0].set_xlabel('Time (s)')
    axes[1, 0].set_ylabel('Y Position (pixels)')
    axes[1, 0].set_title('Y Position')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Velocity plots
    if vel_error_x is not None:
        axes[0, 1].plot(tracked['t'], tracked['vx'], 'b-', label='Tracked', alpha=0.7)
        axes[0, 1].plot(ground_truth['t'], ground_truth['vx'], 'r--', label='Ground Truth', alpha=0.7)
        axes[0, 1].set_xlabel('Time (s)')
        axes[0, 1].set_ylabel('X Velocity (pixels/s)')
        axes[0, 1].set_title('X Velocity')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        axes[1, 1].plot(tracked['t'], tracked['vy'], 'b-', label='Tracked', alpha=0.7)
        axes[1, 1].plot(ground_truth['t'], ground_truth['vy'], 'r--', label='Ground Truth', alpha=0.7)
        axes[1, 1].set_xlabel('Time (s)')
        axes[1, 1].set_ylabel('Y Velocity (pixels/s)')
        axes[1, 1].set_title('Y Velocity')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
    else:
        axes[0, 1].text(0.5, 0.5, 'Velocity data not available', ha='center', va='center')
        axes[1, 1].text(0.5, 0.5, 'Velocity data not available', ha='center', va='center')
    
    # Acceleration plots
    if acc_error_x is not None:
        axes[0, 2].plot(tracked['t'], tracked['ax'], 'b-', label='Tracked', alpha=0.7)
        axes[0, 2].plot(ground_truth['t'], ground_truth['ax'], 'r--', label='Ground Truth', alpha=0.7)
        axes[0, 2].set_xlabel('Time (s)')
        axes[0, 2].set_ylabel('X Acceleration (pixels/s²)')
        axes[0, 2].set_title('X Acceleration')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
        
        axes[1, 2].plot(tracked['t'], tracked['ay'], 'b-', label='Tracked', alpha=0.7)
        axes[1, 2].plot(ground_truth['t'], ground_truth['ay'], 'r--', label='Ground Truth', alpha=0.7)
        axes[1, 2].set_xlabel('Time (s)')
        axes[1, 2].set_ylabel('Y Acceleration (pixels/s²)')
        axes[1, 2].set_title('Y Acceleration')
        axes[1, 2].legend()
        axes[1, 2].grid(True, alpha=0.3)
    else:
        axes[0, 2].text(0.5, 0.5, 'Acceleration data not available', ha='center', va='center')
        axes[1, 2].text(0.5, 0.5, 'Acceleration data not available', ha='center', va='center')
    
    plt.tight_layout()
    plot_path = Path(output_dir) / "tracking_comparison.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"\nComparison plot saved to: {plot_path}")
    
    # Error plots
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('Tracking Errors vs Ground Truth', fontsize=16)
    
    axes[0, 0].plot(tracked['t'], pos_error_x, 'b-', alpha=0.7)
    axes[0, 0].set_xlabel('Time (s)')
    axes[0, 0].set_ylabel('X Position Error (pixels)')
    axes[0, 0].set_title(f'X Position Error (Mean: {pos_error_x.mean():.2f})')
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].plot(tracked['t'], pos_error_y, 'b-', alpha=0.7)
    axes[0, 1].set_xlabel('Time (s)')
    axes[0, 1].set_ylabel('Y Position Error (pixels)')
    axes[0, 1].set_title(f'Y Position Error (Mean: {pos_error_y.mean():.2f})')
    axes[0, 1].grid(True, alpha=0.3)
    
    if vel_error_x is not None:
        axes[1, 0].plot(tracked['t'], vel_error_x, 'r-', alpha=0.7)
        axes[1, 0].set_xlabel('Time (s)')
        axes[1, 0].set_ylabel('X Velocity Error (pixels/s)')
        axes[1, 0].set_title(f'X Velocity Error (Mean: {vel_error_x.mean():.2f})')
        axes[1, 0].grid(True, alpha=0.3)
        
        axes[1, 1].plot(tracked['t'], vel_error_y, 'r-', alpha=0.7)
        axes[1, 1].set_xlabel('Time (s)')
        axes[1, 1].set_ylabel('Y Velocity Error (pixels/s)')
        axes[1, 1].set_title(f'Y Velocity Error (Mean: {vel_error_y.mean():.2f})')
        axes[1, 1].grid(True, alpha=0.3)
    else:
        axes[1, 0].text(0.5, 0.5, 'Velocity error not available', ha='center', va='center')
        axes[1, 1].text(0.5, 0.5, 'Velocity error not available', ha='center', va='center')
    
    plt.tight_layout()
    error_plot_path = Path(output_dir) / "tracking_errors.png"
    plt.savefig(error_plot_path, dpi=300, bbox_inches='tight')
    print(f"Error plot saved to: {error_plot_path}")
    
    return {
        'pos_error_x': pos_error_x,
        'pos_error_y': pos_error_y,
        'vel_error_x': vel_error_x,
        'vel_error_y': vel_error_y,
        'acc_error_x': acc_error_x,
        'acc_error_y': acc_error_y
    }

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) != 3:
        print("Usage: python compare_results.py <tracked_csv> <groundtruth_csv>")
        sys.exit(1)
    
    tracked_file = sys.argv[1]
    groundtruth_file = sys.argv[2]
    
    errors = compare_tracking_results(tracked_file, groundtruth_file)
    
    print("\n=== SUMMARY ===")
    print("✓ Tracking comparison completed successfully!")
    print("✓ Check the comparison_plots/ directory for visualizations")
    print("✓ Inspect outputs/overlay.mp4 for visual tracking quality")
