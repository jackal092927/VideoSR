#!/usr/bin/env python3
"""
Generate synthetic noisy trajectory data for testing physics recovery.
Creates data that follows a smooth equation (e.g., projectile motion) 
with random perturbations to simulate real-world noise.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt


def generate_noisy_trajectory(n_frames=120, fps=30.0, noise_level=0.1, perturbation_prob=0.3):
    """
    Generate synthetic noisy trajectory data.
    
    Args:
        n_frames: Number of frames
        fps: Frames per second
        noise_level: Standard deviation of Gaussian noise (pixels)
        perturbation_prob: Probability of adding random perturbation per frame
    
    Returns:
        DataFrame with columns: [frame, t, x, y, vx, vy, ax, ay, area, ok]
    """
    
    # Time array
    t = np.arange(n_frames) / fps
    
    # Ideal projectile motion parameters (smooth trajectory)
    g = 9.8  # m/s^2, scaled for pixels
    v0x = 100.0  # initial horizontal velocity (pixels/s)
    v0y = -150.0  # initial vertical velocity (pixels/s)
    x0 = 50.0  # initial x position
    y0 = 400.0  # initial y position
    
    # Ideal smooth trajectory (no noise)
    x_ideal = x0 + v0x * t
    y_ideal = y0 + v0y * t + 0.5 * g * t**2
    
    # Velocities and accelerations
    vx_ideal = v0x * np.ones_like(t)
    vy_ideal = v0y + g * t
    ax_ideal = np.zeros_like(t)
    ay_ideal = g * np.ones_like(t)
    
    # Add noise and perturbations
    x_noisy = x_ideal.copy()
    y_noisy = y_ideal.copy()
    
    # 1. Add small Gaussian noise to all frames
    x_noisy += np.random.normal(0, noise_level, n_frames)
    y_noisy += np.random.normal(0, noise_level, n_frames)
    
    # 2. Add random perturbations (larger jumps) to some frames
    for i in range(n_frames):
        if np.random.random() < perturbation_prob:
            # Add larger perturbation
            perturbation = np.random.normal(0, noise_level * 5, 2)
            x_noisy[i] += perturbation[0]
            y_noisy[i] += perturbation[1]
    
    # 3. Add some outliers (completely wrong detections)
    outlier_frames = np.random.choice(n_frames, size=int(n_frames * 0.05), replace=False)
    for frame_idx in outlier_frames:
        x_noisy[frame_idx] = np.random.uniform(0, 800)  # random x
        y_noisy[frame_idx] = np.random.uniform(0, 600)  # random y
    
    # Calculate noisy velocities and accelerations
    vx_noisy = np.gradient(x_noisy, t)
    vy_noisy = np.gradient(y_noisy, t)
    ax_noisy = np.gradient(vx_noisy, t)
    ay_noisy = np.gradient(vy_noisy, t)
    
    # Create DataFrame
    data = {
        'frame': np.arange(n_frames),
        't': t,
        'x': x_noisy,
        'y': y_noisy,
        'vx': vx_noisy,
        'vy': vy_noisy,
        'ax': ax_noisy,
        'ay': ay_noisy,
        'area': np.random.uniform(80, 120, n_frames),  # simulated area
        'ok': np.ones(n_frames)  # all frames are "ok"
    }
    
    df = pd.DataFrame(data)
    
    # Mark some frames as "not ok" for outliers
    df.loc[outlier_frames, 'ok'] = 0
    
    return df, {
        'x_ideal': x_ideal,
        'y_ideal': y_ideal,
        'vx_ideal': vx_ideal,
        'vy_ideal': vy_ideal,
        'ax_ideal': ax_ideal,
        'ay_ideal': ay_ideal
    }


def plot_trajectory_comparison(df, ideal_data, save_path=None):
    """Plot noisy vs ideal trajectory for comparison."""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Position plots
    axes[0, 0].plot(df['x'], df['y'], 'b.', alpha=0.6, label='Noisy Data')
    axes[0, 0].plot(ideal_data['x_ideal'], ideal_data['y_ideal'], 'r-', linewidth=2, label='Ideal Trajectory')
    axes[0, 0].set_xlabel('X Position (pixels)')
    axes[0, 0].set_ylabel('Y Position (pixels)')
    axes[0, 0].set_title('Trajectory Comparison')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Time series plots
    axes[0, 1].plot(df['t'], df['x'], 'b.', alpha=0.6, label='Noisy X')
    axes[0, 1].plot(df['t'], ideal_data['x_ideal'], 'r-', linewidth=2, label='Ideal X')
    axes[0, 1].set_xlabel('Time (s)')
    axes[0, 1].set_ylabel('X Position (pixels)')
    axes[0, 1].set_title('X Position vs Time')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[1, 0].plot(df['t'], df['y'], 'b.', alpha=0.6, label='Noisy Y')
    axes[1, 0].plot(df['t'], ideal_data['y_ideal'], 'r-', linewidth=2, label='Ideal Y')
    axes[1, 0].set_xlabel('Time (s)')
    axes[1, 0].set_ylabel('Y Position (pixels)')
    axes[1, 0].set_title('Y Position vs Time')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Velocity plot
    axes[1, 1].plot(df['t'], df['vy'], 'b.', alpha=0.6, label='Noisy Vy')
    axes[1, 1].plot(df['t'], ideal_data['vy_ideal'], 'r-', linewidth=2, label='Ideal Vy')
    axes[1, 1].set_xlabel('Time (s)')
    axes[1, 1].set_ylabel('Y Velocity (pixels/s)')
    axes[1, 1].set_title('Y Velocity vs Time')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    
    plt.show()


def main():
    """Generate and save noisy trajectory data."""
    
    # Create outputs directory
    Path("outputs").mkdir(parents=True, exist_ok=True)
    
    # Generate noisy data
    print("Generating synthetic noisy trajectory data...")
    df, ideal_data = generate_noisy_trajectory(
        n_frames=120,
        fps=30.0,
        noise_level=2.0,  # 2 pixels of noise
        perturbation_prob=0.2  # 20% chance of perturbation per frame
    )
    
    # Save noisy data
    csv_path = "outputs/synthetic_noisy_trajectory.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved noisy data to: {csv_path}")
    
    # Save ideal trajectory data for comparison
    ideal_df = pd.DataFrame({
        'frame': np.arange(len(ideal_data['x_ideal'])),
        't': df['t'],
        'x': ideal_data['x_ideal'],
        'y': ideal_data['y_ideal'],
        'vx': ideal_data['vx_ideal'],
        'vy': ideal_data['vy_ideal'],
        'ax': ideal_data['ax_ideal'],
        'ay': ideal_data['ay_ideal'],
        'area': df['area'],
        'ok': np.ones(len(ideal_data['x_ideal']))
    })
    
    ideal_csv_path = "outputs/synthetic_ideal_trajectory.csv"
    ideal_df.to_csv(ideal_csv_path, index=False)
    print(f"Saved ideal trajectory data to: {ideal_csv_path}")
    
    print(f"Data shape: {df.shape}")
    print(f"Outlier frames (ok=0): {len(df[df['ok'] == 0])}")
    
    # Plot comparison
    plot_path = "outputs/trajectory_comparison.png"
    plot_trajectory_comparison(df, ideal_data, save_path=plot_path)
    
    # Print some statistics
    print("\nData Statistics:")
    print(f"X range: {df['x'].min():.1f} to {df['x'].max():.1f}")
    print(f"Y range: {df['y'].min():.1f} to {df['y'].max():.1f}")
    print(f"X velocity range: {df['vx'].min():.1f} to {df['vx'].max():.1f}")
    print(f"Y velocity range: {df['vy'].min():.1f} to {df['vy'].max():.1f}")
    
    print("\nIdeal vs Noisy Comparison:")
    print(f"X RMSE: {np.sqrt(np.mean((df['x'] - ideal_data['x_ideal'])**2)):.2f} pixels")
    print(f"Y RMSE: {np.sqrt(np.mean((df['y'] - ideal_data['y_ideal'])**2)):.2f} pixels")


if __name__ == "__main__":
    main()
