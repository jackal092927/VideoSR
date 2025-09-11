#!/usr/bin/env python3
"""
Generate reconstructed trajectory videos from discovered physics equations.
This script creates three types of videos:
1. Standalone reconstructed trajectory video
2. Side-by-side comparison: Real vs Reconstructed
3. All videos for comprehensive analysis
"""

import subprocess
from pathlib import Path


def run_command(cmd, description):
    """Run a command and show progress."""
    print(f"\n🎬 {description}")
    print(f"Command: {cmd}")
    print("-" * 50)
    
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        print("✅ Success!")
        if result.stdout:
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error: {e}")
        if e.stdout:
            print("STDOUT:", e.stdout)
        if e.stderr:
            print("STDERR:", e.stderr)
        return False


def main():
    """Generate all reconstructed trajectory videos."""
    
    print("🎯 GENERATING RECONSTRUCTED TRAJECTORY VIDEOS")
    print("=" * 60)
    
    # Check what tracking data and equations we have
    outputs_dir = Path("outputs")
    track_files = list(outputs_dir.glob("track_*.csv"))
    equations_files = list(outputs_dir.glob("track_*.equations.txt"))
    
    if not track_files:
        print("❌ Error: No tracking data files found!")
        print("Please run the tracking pipeline first to get trajectory data.")
        return
    
    if not equations_files:
        print("❌ Error: No equations files found!")
        print("Please run the physics discovery pipeline first.")
        return
    
    # Use the latest tracking data and equations
    latest_track = max(track_files, key=lambda x: x.stat().st_mtime)
    latest_equations = max(equations_files, key=lambda x: x.stat().st_mtime)
    
    print(f"📊 Using latest tracking data: {latest_track.name}")
    print(f"📊 Using latest equations: {latest_equations.name}")
    
    # Read and display the discovered equations
    with open(latest_equations, 'r') as f:
        equations_text = f.read().strip()
    
    print(f"\n🔬 Discovered Physics Equations:")
    print("-" * 40)
    for i, line in enumerate(equations_text.split('\n')):
        print(f"  Equation {i+1}: {line}")
    
    # Create videos directory
    videos_dir = Path("outputs/videos")
    videos_dir.mkdir(exist_ok=True)
    
    print(f"\n📁 Videos will be saved to: {videos_dir}")
    
    # 1. Generate standalone reconstructed trajectory video
    print(f"\n🎯 STEP 1: Generating standalone reconstructed trajectory video")
    cmd1 = f"python generate_video_from_csv.py --type reconstructed --equations {latest_equations} --output {videos_dir}/reconstructed_trajectory.mp4 --width 800 --height 600 --fps 30 --duration 4.0"
    if not run_command(cmd1, "Generating reconstructed trajectory from equations"):
        return
    
    # 2. Generate side-by-side comparison: Real vs Reconstructed
    print(f"\n🎯 STEP 2: Generating side-by-side comparison video")
    cmd2 = f"python generate_video_from_csv.py --type real_vs_reconstructed --csv {latest_track} --equations {latest_equations} --output {videos_dir}/real_vs_reconstructed.mp4 --width 800 --height 600 --fps 30"
    if not run_command(cmd2, "Generating real vs reconstructed comparison"):
        return
    
    # 3. Generate additional analysis videos
    print(f"\n🎯 STEP 3: Generating additional analysis videos")
    
    # Real trajectory with equations overlay
    cmd3 = f"python generate_video_from_csv.py --csv {latest_track} --output {videos_dir}/real_with_equations.mp4 --width 800 --height 600 --fps 30 --equations {latest_equations}"
    if not run_command(cmd3, "Generating real trajectory with equations overlay"):
        return
    
    # Real trajectory with velocity vectors
    cmd4 = f"python generate_video_from_csv.py --csv {latest_track} --output {videos_dir}/real_with_velocity.mp4 --width 800 --height 600 --fps 30 --show-velocity"
    if not run_command(cmd4, "Generating real trajectory with velocity vectors"):
        return
    
    # Summary
    print(f"\n🎉 ALL RECONSTRUCTED TRAJECTORY VIDEOS GENERATED SUCCESSFULLY!")
    print("=" * 60)
    
    video_files = list(videos_dir.glob("*.mp4"))
    if video_files:
        print(f"\n📁 Generated videos ({len(video_files)} files):")
        for video_file in sorted(video_files):
            size_mb = video_file.stat().st_size / (1024 * 1024)
            print(f"  • {video_file.name} ({size_mb:.1f} MB)")
    
    print(f"\n🔍 What each video shows:")
    print(f"  • reconstructed_trajectory.mp4 - Trajectory generated ONLY from discovered equations")
    print(f"  • real_vs_reconstructed.mp4 - Side-by-side comparison of real vs reconstructed")
    print(f"  • real_with_equations.mp4 - Real trajectory with equations displayed")
    print(f"  • real_with_velocity.mp4 - Real trajectory with velocity vectors")
    
    print(f"\n💡 Key insights you can now see:")
    print(f"  • How well the discovered equations can reconstruct the motion")
    print(f"  • Whether the physics discovery captured the essential dynamics")
    print(f"  • How much the reconstructed trajectory differs from the real one")
    print(f"  • Whether the equations represent a good model of the system")
    
    print(f"\n📊 Analysis questions to answer:")
    print(f"  • Does the reconstructed trajectory follow a similar path?")
    print(f"  • Are the velocity patterns similar?")
    print(f"  • How much error exists between real and reconstructed?")
    print(f"  • Do the equations capture the fundamental physics?")
    
    # Show equations again for reference
    print(f"\n🔬 Equations used for reconstruction:")
    print("-" * 40)
    for i, line in enumerate(equations_text.split('\n')):
        print(f"  {i+1}. {line}")
    
    print(f"\n🎯 The reconstructed trajectory video shows what motion would look like")
    print(f"   if an object perfectly followed your discovered physics equations!")
    print(f"   Compare it with the real trajectory to see how well you captured the physics.")


if __name__ == "__main__":
    main()
