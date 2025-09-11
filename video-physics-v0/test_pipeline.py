#!/usr/bin/env python3
"""
Test the complete pipeline: generate noisy data and test physics recovery.
This script demonstrates how well the method can recover underlying physics
from noisy, perturbed trajectory data.
"""

import subprocess
import sys
from pathlib import Path


def run_command(cmd, description):
    """Run a command and handle errors."""
    print(f"\n{'='*60}")
    print(f"STEP: {description}")
    print(f"COMMAND: {cmd}")
    print('='*60)
    
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        return True
    except subprocess.CalledProcessError as e:
        print(f"ERROR: Command failed with exit code {e.returncode}")
        print("STDOUT:", e.stdout)
        print("STDERR:", e.stderr)
        return False


def main():
    """Run the complete test pipeline."""
    
    print("🧪 TESTING PHYSICS RECOVERY ON NOISY DATA")
    print("This will test how well the method can recover underlying physics from noisy trajectories.")
    
    # Check if we're in the right directory
    if not Path("main.py").exists():
        print("❌ Error: Please run this script from the video-physics-v0 directory")
        sys.exit(1)
    
    # Step 1: Generate synthetic noisy data
    if not run_command("python generate_noisy_data.py", "Generate synthetic noisy trajectory data"):
        print("❌ Failed to generate synthetic data")
        sys.exit(1)
    
    # Step 2: Test physics recovery
    if not run_command("python test_noisy_recovery.py", "Test physics recovery on noisy data"):
        print("❌ Failed to test physics recovery")
        sys.exit(1)
    
    # Step 3: Show results summary
    print(f"\n{'='*60}")
    print("🎉 PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*60)
    
    # Check what files were created
    output_files = list(Path("outputs").glob("*"))
    if output_files:
        print("\n📁 Generated output files:")
        for file_path in sorted(output_files):
            size = file_path.stat().st_size
            if size > 1024:
                size_str = f"{size/1024:.1f} KB"
            else:
                size_str = f"{size} B"
            print(f"  • {file_path.name} ({size_str})")
    
    print("\n📊 Key results:")
    print("  • Synthetic noisy trajectory data generated")
    print("  • Physics equations discovered using SINDy")
    print("  • Recovery quality analyzed and plotted")
    print("  • Comparison with ideal trajectory completed")
    
    print("\n🔍 To view results:")
    print("  • Check the plots in the outputs/ directory")
    print("  • Review recovery_summary.txt for detailed results")
    print("  • Compare noisy vs ideal trajectory data")
    
    print("\n💡 The test demonstrates:")
    print("  • How well the method handles noisy data")
    print("  • Whether underlying physics can be recovered")
    print("  • Quality of trajectory reconstruction")
    print("  • Robustness to perturbations and outliers")


if __name__ == "__main__":
    main()

