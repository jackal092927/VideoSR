#!/usr/bin/env python3
"""
Test physics recovery on synthetic noisy trajectory data.
This script tests how well the method can recover the underlying physics
from noisy, perturbed data.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from discovery.sindy_fit import run_sindy


def test_physics_recovery(csv_path, config=None):
    """
    Test physics recovery on noisy trajectory data.
    
    Args:
        csv_path: Path to CSV file with noisy trajectory data
        config: Configuration dictionary for SINDy
    
    Returns:
        Dictionary with results and comparisons
    """
    
    # Load noisy data
    print(f"Loading noisy trajectory data from: {csv_path}")
    df = pd.read_csv(csv_path)
    
    # Load ideal trajectory data (if available)
    ideal_data = None
    ideal_csv = csv_path.replace('_noisy_', '_ideal_')
    if Path(ideal_csv).exists():
        ideal_df = pd.read_csv(ideal_csv)
        ideal_data = {
            'x_ideal': ideal_df['x'].values,
            'y_ideal': ideal_df['y'].values,
            'vx_ideal': ideal_df['vx'].values,
            'vy_ideal': ideal_df['vy'].values,
            'ax_ideal': ideal_df['ax'].values,
            'ay_ideal': ideal_df['ay'].values
        }
    
    # Default SINDy configuration
    if config is None:
        config = {
            'poly_degree': 3,
            'thresh': 0.01,
            'discrete_time': False
        }
    
    # Run SINDy to discover physics equations
    print("\nRunning SINDy to discover physics equations...")
    try:
        eqn_txt = run_sindy(df, config)
        print("=== Discovered Equations ===")
        print(eqn_txt)
        
        # Parse equations
        equations = eqn_txt.strip().split('\n')
        
    except Exception as e:
        print(f"Error running SINDy: {e}")
        equations = []
        eqn_txt = ""
    
    # Analyze data quality and recovery
    results = {
        'data_shape': df.shape,
        'noise_level': estimate_noise_level(df),
        'outlier_frames': len(df[df['ok'] == 0]) if 'ok' in df.columns else 0,
        'equations': equations,
        'equation_text': eqn_txt,
        'config': config
    }
    
    # Compare with ideal trajectory if available
    if ideal_data is not None:
        results['ideal_comparison'] = compare_with_ideal(df, ideal_data)
    
    return results, df, ideal_data


def estimate_noise_level(df):
    """Estimate noise level in the data using simple statistics."""
    
    # Use velocity data to estimate noise (should be smoother than position)
    if 'vx' in df.columns and 'vy' in df.columns:
        # Calculate local variance in velocities
        vx_var = np.var(df['vx'].rolling(window=5, center=True).mean().dropna())
        vy_var = np.var(df['vy'].rolling(window=5, center=True).mean().dropna())
        return np.sqrt(vx_var + vy_var)
    
    return None


def compare_with_ideal(df, ideal_data):
    """Compare recovered data with ideal trajectory."""
    
    comparison = {}
    
    # Position RMSE
    if 'x' in df.columns and 'x_ideal' in ideal_data:
        x_rmse = np.sqrt(np.mean((df['x'] - ideal_data['x_ideal'])**2))
        y_rmse = np.sqrt(np.mean((df['y'] - ideal_data['y_ideal'])**2))
        comparison['position_rmse'] = {'x': x_rmse, 'y': y_rmse, 'total': np.sqrt(x_rmse**2 + y_rmse**2)}
    
    # Velocity RMSE
    if 'vx' in df.columns and 'vx_ideal' in ideal_data:
        vx_rmse = np.sqrt(np.mean((df['vx'] - ideal_data['vx_ideal'])**2))
        vy_rmse = np.sqrt(np.mean((df['vy'] - ideal_data['vy_ideal'])**2))
        comparison['velocity_rmse'] = {'x': vx_rmse, 'y': vy_rmse, 'total': np.sqrt(vx_rmse**2 + vy_rmse**2)}
    
    # Acceleration RMSE
    if 'ax' in df.columns and 'ax_ideal' in ideal_data:
        ax_rmse = np.sqrt(np.mean((df['ax'] - ideal_data['ax_ideal'])**2))
        ay_rmse = np.sqrt(np.mean((df['ay'] - ideal_data['ay_ideal'])**2))
        comparison['acceleration_rmse'] = {'x': ax_rmse, 'y': ay_rmse, 'total': np.sqrt(ax_rmse**2 + ay_rmse**2)}
    
    return comparison


def plot_recovery_results(df, ideal_data, results, save_path=None):
    """Plot the recovery results showing noisy data, ideal trajectory, and discovered equations."""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Trajectory comparison
    axes[0, 0].plot(df['x'], df['y'], 'b.', alpha=0.6, markersize=3, label='Noisy Data')
    if ideal_data:
        axes[0, 0].plot(ideal_data['x_ideal'], ideal_data['y_ideal'], 'r-', linewidth=2, label='Ideal Trajectory')
    axes[0, 0].set_xlabel('X Position (pixels)')
    axes[0, 0].set_ylabel('Y Position (pixels)')
    axes[0, 0].set_title('Trajectory Recovery')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # X position vs time
    axes[0, 1].plot(df['t'], df['x'], 'b.', alpha=0.6, markersize=3, label='Noisy X')
    if ideal_data:
        axes[0, 1].plot(df['t'], ideal_data['x_ideal'], 'r-', linewidth=2, label='Ideal X')
    axes[0, 1].set_xlabel('Time (s)')
    axes[0, 1].set_ylabel('X Position (pixels)')
    axes[0, 1].set_title('X Position Recovery')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Y position vs time
    axes[0, 2].plot(df['t'], df['y'], 'b.', alpha=0.6, markersize=3, label='Noisy Y')
    if ideal_data:
        axes[0, 2].plot(df['t'], ideal_data['y_ideal'], 'r-', linewidth=2, label='Ideal Y')
    axes[0, 2].set_xlabel('Time (s)')
    axes[0, 2].set_ylabel('Y Position (pixels)')
    axes[0, 2].set_title('Y Position Recovery')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    # Y velocity vs time
    axes[1, 0].plot(df['t'], df['vy'], 'b.', alpha=0.6, markersize=3, label='Noisy Vy')
    if ideal_data:
        axes[1, 0].plot(df['t'], ideal_data['vy_ideal'], 'r-', linewidth=2, label='Ideal Vy')
    axes[1, 0].set_xlabel('Time (s)')
    axes[1, 0].set_ylabel('Y Velocity (pixels/s)')
    axes[1, 0].set_title('Y Velocity Recovery')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Data quality indicators
    if 'ok' in df.columns:
        ok_frames = df[df['ok'] == 1]
        bad_frames = df[df['ok'] == 0]
        
        axes[1, 1].plot(ok_frames['t'], ok_frames['x'], 'g.', markersize=3, label='Good Frames')
        if len(bad_frames) > 0:
            axes[1, 1].plot(bad_frames['t'], bad_frames['x'], 'r.', markersize=5, label='Outlier Frames')
        axes[1, 1].set_xlabel('Time (s)')
    else:
        axes[1, 1].plot(df['t'], df['x'], 'b.', markersize=3)
        axes[1, 1].set_xlabel('Time (s)')
    
    axes[1, 1].set_ylabel('X Position (pixels)')
    axes[1, 1].set_title('Data Quality (X)')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # Discovered equations
    axes[1, 2].text(0.1, 0.9, 'Discovered Equations:', transform=axes[1, 2].transAxes, 
                     fontsize=12, fontweight='bold')
    
    if results['equations']:
        y_pos = 0.8
        for i, eq in enumerate(results['equations'][:5]):  # Show first 5 equations
            if y_pos > 0.1:
                axes[1, 2].text(0.1, y_pos, f"{i+1}. {eq}", transform=axes[1, 2].transAxes, 
                               fontsize=10, fontfamily='monospace')
                y_pos -= 0.15
    else:
        axes[1, 2].text(0.1, 0.5, 'No equations discovered', transform=axes[1, 2].transAxes, 
                        fontsize=12, style='italic')
    
    axes[1, 2].set_xlim(0, 1)
    axes[1, 2].set_ylim(0, 1)
    axes[1, 2].set_title('Physics Discovery')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Recovery results plot saved to: {save_path}")
    
    plt.show()


def main():
    """Main function to test physics recovery on noisy data."""
    
    # Check if synthetic data exists
    csv_path = "outputs/synthetic_noisy_trajectory.csv"
    
    if not Path(csv_path).exists():
        print(f"Error: {csv_path} not found!")
        print("Please run generate_noisy_data.py first to create synthetic data.")
        return
    
    # Test physics recovery
    print("Testing physics recovery on synthetic noisy data...")
    results, df, ideal_data = test_physics_recovery(csv_path)
    
    # Print results summary
    print("\n" + "="*50)
    print("PHYSICS RECOVERY RESULTS SUMMARY")
    print("="*50)
    print(f"Data shape: {results['data_shape']}")
    print(f"Estimated noise level: {results['noise_level']:.2f}" if results['noise_level'] else "Noise level: Unknown")
    print(f"Outlier frames: {results['outlier_frames']}")
    print(f"Equations discovered: {len(results['equations'])}")
    
    if ideal_data and 'ideal_comparison' in results:
        comp = results['ideal_comparison']
        print("\nRecovery Quality (RMSE):")
        if 'position_rmse' in comp:
            print(f"  Position: X={comp['position_rmse']['x']:.2f}, Y={comp['position_rmse']['y']:.2f}, Total={comp['position_rmse']['total']:.2f}")
        if 'velocity_rmse' in comp:
            print(f"  Velocity: X={comp['velocity_rmse']['x']:.2f}, Y={comp['velocity_rmse']['y']:.2f}, Total={comp['velocity_rmse']['total']:.2f}")
        if 'acceleration_rmse' in comp:
            print(f"  Acceleration: X={comp['acceleration_rmse']['x']:.2f}, Y={comp['acceleration_rmse']['y']:.2f}, Total={comp['acceleration_rmse']['total']:.2f}")
    
    # Plot results
    plot_path = "outputs/recovery_results.png"
    plot_recovery_results(df, ideal_data, results, save_path=plot_path)
    
    # Save results summary
    summary_path = "outputs/recovery_summary.txt"
    with open(summary_path, 'w') as f:
        f.write("PHYSICS RECOVERY TEST RESULTS\n")
        f.write("="*50 + "\n\n")
        f.write(f"Data file: {csv_path}\n")
        f.write(f"Data shape: {results['data_shape']}\n")
        f.write(f"Outlier frames: {results['outlier_frames']}\n")
        f.write(f"Equations discovered: {len(results['equations'])}\n\n")
        f.write("Discovered Equations:\n")
        for i, eq in enumerate(results['equations']):
            f.write(f"{i+1}. {eq}\n")
        f.write("\n" + "="*50 + "\n")
    
    print(f"\nResults summary saved to: {summary_path}")


if __name__ == "__main__":
    main()

