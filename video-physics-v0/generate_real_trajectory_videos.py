#!/usr/bin/env python3
"""
Generate videos from real trajectory tracking data.
This script creates videos showing the actual curved trajectories
from your video tracking results.
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
    """Generate videos from real tracking data."""
    
    print("🎥 GENERATING VIDEOS FROM REAL TRAJECTORY TRACKING DATA")
    print("=" * 60)
    
    # Check what tracking data files we have
    outputs_dir = Path("outputs")
    track_files = list(outputs_dir.glob("track_*.csv"))
    
    if not track_files:
        print("❌ Error: No tracking data files found!")
        print("Please run the tracking pipeline first to get trajectory data.")
        return
    
    # Use the latest tracking data
    latest_track = max(track_files, key=lambda x: x.stat().st_mtime)
    print(f"📊 Using latest tracking data: {latest_track.name}")
    
    # Check if we have equations file
    equations_file = latest_track.with_suffix(".equations.txt")
    if not equations_file.exists():
        print(f"⚠️  No equations file found for {latest_track.name}")
        equations_file = None
    
    # Create videos directory
    videos_dir = Path("outputs/videos")
    videos_dir.mkdir(exist_ok=True)
    
    print(f"📁 Videos will be saved to: {videos_dir}")
    
    # 1. Generate real trajectory video
    cmd1 = f"python generate_video_from_csv.py --csv {latest_track} --output {videos_dir}/real_trajectory.mp4 --width 800 --height 600 --fps 30"
    if not run_command(cmd1, "Generating real trajectory video"):
        return
    
    # 2. Generate trajectory video with equations (if available)
    if equations_file and equations_file.exists():
        cmd2 = f"python generate_video_from_csv.py --csv {latest_track} --output {videos_dir}/real_with_equations.mp4 --width 800 --height 600 --fps 30 --equations {equations_file}"
        if not run_command(cmd2, "Generating real trajectory video with discovered equations"):
            return
    else:
        print("⚠️  No equations file found, skipping equation overlay video")
    
    # 3. Generate velocity visualization video
    cmd3 = f"python generate_video_from_csv.py --csv {latest_track} --output {videos_dir}/real_with_velocity.mp4 --width 800 --height 600 --fps 30 --show-velocity"
    if not run_command(cmd3, "Generating real trajectory video with velocity vectors"):
        return
    
    # 4. Generate acceleration visualization video (if we modify the script to support it)
    # For now, just show what we have
    
    # Summary
    print(f"\n🎉 REAL TRAJECTORY VIDEOS GENERATED SUCCESSFULLY!")
    print("=" * 60)
    
    video_files = list(videos_dir.glob("*.mp4"))
    if video_files:
        print(f"\n📁 Generated videos ({len(video_files)} files):")
        for video_file in sorted(video_files):
            size_mb = video_file.stat().st_size / (1024 * 1024)
            print(f"  • {video_file.name} ({size_mb:.1f} MB)")
    
    print(f"\n🔍 What each video shows:")
    print(f"  • real_trajectory.mp4 - Actual curved trajectory from your video tracking")
    print(f"  • real_with_equations.mp4 - Real trajectory with discovered physics equations")
    print(f"  • real_with_velocity.mp4 - Real trajectory with velocity vector visualization")
    
    print(f"\n💡 These videos show:")
    print(f"  • The actual curved path of the tracked object")
    print(f"  • How well the physics equations fit the real data")
    print(f"  • Velocity and acceleration patterns in the real trajectory")
    
    print(f"\n📊 You can now visually assess:")
    print(f"  • Quality of trajectory tracking from your video")
    print(f"  • How well the discovered physics equations match real motion")
    print(f"  • Whether the tracking captured the expected curved path")
    
    # Show some info about the tracking data
    print(f"\n📈 Tracking data info:")
    print(f"  • Source file: {latest_track.name}")
    print(f"  • File size: {latest_track.stat().st_size / 1024:.1f} KB")
    if equations_file and equations_file.exists():
        print(f"  • Equations file: {equations_file.name}")
        with open(equations_file, 'r') as f:
            equations = f.read().strip()
            print(f"  • Discovered equations: {equations}")


if __name__ == "__main__":
    main()
