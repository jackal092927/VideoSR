#!/usr/bin/env python3
"""
Generate all videos from the synthetic trajectory data.
This script creates videos showing the noisy vs ideal trajectories
and the physics recovery results.
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
    """Generate all videos from the synthetic data."""
    
    print("🎥 GENERATING VIDEOS FROM SYNTHETIC TRAJECTORY DATA")
    print("=" * 60)
    
    # Check if required files exist
    noisy_csv = "outputs/synthetic_noisy_trajectory.csv"
    ideal_csv = "outputs/synthetic_ideal_trajectory.csv"
    equations_txt = "outputs/track_20250827_050834.equations.txt"
    
    if not Path(noisy_csv).exists():
        print(f"❌ Error: {noisy_csv} not found!")
        print("Please run generate_noisy_data.py first.")
        return
    
    if not Path(ideal_csv).exists():
        print(f"❌ Error: {ideal_csv} not found!")
        print("Please run generate_noisy_data.py first.")
        return
    
    # Create videos directory
    videos_dir = Path("outputs/videos")
    videos_dir.mkdir(exist_ok=True)
    
    print(f"📁 Videos will be saved to: {videos_dir}")
    
    # 1. Generate noisy trajectory video
    cmd1 = f"python generate_video_from_csv.py --csv {noisy_csv} --output {videos_dir}/noisy_trajectory.mp4 --width 800 --height 600 --fps 30"
    if not run_command(cmd1, "Generating noisy trajectory video"):
        return
    
    # 2. Generate ideal trajectory video
    cmd2 = f"python generate_video_from_csv.py --csv {ideal_csv} --output {videos_dir}/ideal_trajectory.mp4 --width 800 --height 600 --fps 30"
    if not run_command(cmd2, "Generating ideal trajectory video"):
        return
    
    # 3. Generate side-by-side comparison video
    cmd3 = f"python generate_video_from_csv.py --type comparison --csv {noisy_csv} --ideal-csv {ideal_csv} --output {videos_dir}/comparison.mp4 --width 800 --height 600 --fps 30"
    if not run_command(cmd3, "Generating side-by-side comparison video"):
        return
    
    # 4. Generate noisy trajectory video with equations (if available)
    if Path(equations_txt).exists():
        cmd4 = f"python generate_video_from_csv.py --csv {noisy_csv} --output {videos_dir}/noisy_with_equations.mp4 --width 800 --height 600 --fps 30 --equations {equations_txt}"
        if not run_command(cmd4, "Generating noisy trajectory video with discovered equations"):
            return
    else:
        print("⚠️  No equations file found, skipping equation overlay video")
    
    # 5. Generate velocity visualization video
    cmd5 = f"python generate_video_from_csv.py --csv {noisy_csv} --output {videos_dir}/noisy_with_velocity.mp4 --width 800 --height 600 --fps 30 --show-velocity"
    if not run_command(cmd5, "Generating noisy trajectory video with velocity vectors"):
        return
    
    # Summary
    print(f"\n🎉 ALL VIDEOS GENERATED SUCCESSFULLY!")
    print("=" * 60)
    
    video_files = list(videos_dir.glob("*.mp4"))
    if video_files:
        print(f"\n📁 Generated videos ({len(video_files)} files):")
        for video_file in sorted(video_files):
            size_mb = video_file.stat().st_size / (1024 * 1024)
            print(f"  • {video_file.name} ({size_mb:.1f} MB)")
    
    print(f"\n🔍 What each video shows:")
    print(f"  • noisy_trajectory.mp4 - Real tracked trajectory from video")
    print(f"  • ideal_trajectory.mp4 - Clean, smooth trajectory (ground truth)")
    print(f"  • comparison.mp4 - Side-by-side comparison of real vs ideal")
    print(f"  • noisy_with_equations.mp4 - Real trajectory with discovered physics equations")
    print(f"  • noisy_with_velocity.mp4 - Real trajectory with velocity vector visualization")
    
    print(f"\n💡 The comparison video is particularly useful for seeing:")
    print(f"  • How well the real tracking data matches the ideal physics")
    print(f"  • Where tracking errors or noise occur")
    print(f"  • The difference between real-world data and ideal trajectories")
    
    print(f"\n📊 You can now visually assess:")
    print(f"  • Quality of real trajectory tracking")
    print(f"  • How well physics equations fit the real data")
    print(f"  • Effectiveness of physics equation discovery on real trajectories")


if __name__ == "__main__":
    main()
