# Video Physics v0

A Python-based video analysis tool for extracting physical dynamics from video sequences using computer vision and symbolic regression.

## Overview

This project combines OpenCV-based object tracking with symbolic regression techniques (PySINDy and optionally PySR) to discover underlying physical laws from video data.

## Features

- **Object Tracking**: OpenCV HSV mask-based centroid tracking for robust object detection
- **Trajectory Analysis**: Smoothing, derivative calculation, and gap filling for clean trajectory data
- **Physics Discovery**: PySINDy integration for symbolic regression of dynamical systems
- **Video Overlay**: Visualization of tracks and trajectories with video output
- **Flexible Modes**: Support for both lightweight (OpenCV) and advanced (OpenVINO) processing modes

## Environment Setup

This project is designed to run with the `video-sr` conda environment. The environment contains all necessary dependencies.

### Using the Run Script (Recommended)

```bash
# Make sure you're in the video-physics-v0 directory
./run.sh input_video.mp4

# Or specify a custom config file
./run.sh input_video.mp4 configs/example.yaml
```

### Manual Execution

```bash
# Activate the conda environment
conda activate video-sr

# Run the project
python main.py --video input_video.mp4 --config configs/example.yaml

# Or use the full path directly
/common/home/cx122/miniconda3/envs/video-sr/bin/python main.py --video input_video.mp4 --config configs/example.yaml
```

## Installation

The project dependencies are managed through the `video-sr` conda environment. If you need to recreate the environment:

```bash
# Create the environment (if it doesn't exist)
conda create -n video-sr python=3.11

# Activate the environment
conda activate video-sr

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Basic Usage

```bash
# Using the run script
./run.sh your_video.mp4

# Using Python directly
python main.py --video your_video.mp4
```

### Configuration

Edit `configs/example.yaml` to customize:
- Input video path
- Tracking parameters (HSV color ranges, area thresholds)
- Output settings
- Analysis modes (PySINDy parameters)

## Project Structure

- `main.py`: Main entry point and orchestration
- `run.sh`: Convenient script to run with the video-sr environment
- `extractor/`: Video processing and object tracking modules
- `discovery/`: Symbolic regression and physics discovery modules
- `configs/`: Configuration files for different analysis scenarios
- `outputs/`: Generated output files (created automatically)

## Dependencies

The following packages are available in the `video-sr` conda environment:
- opencv-python
- numpy
- pandas
- scipy
- matplotlib
- pysindy
- pyyaml

## Output Files

- `outputs/track.csv`: Trajectory data with positions, velocities, and accelerations
- `outputs/overlay.mp4`: Video with tracking overlay
- `outputs/track.equations.txt`: Discovered physics equations (if PySINDy is enabled)

## Troubleshooting

### Conda Environment Issues
- Ensure the `video-sr` environment exists: `conda env list`
- Activate the environment: `conda activate video-sr`
- Check Python path: `which python` should point to the conda environment

### Package Import Issues
- Verify all packages are installed: `python -c "import cv2, numpy, pandas, scipy, matplotlib, pysindy, yaml; print('OK')"`
- Reinstall packages if needed: `pip install -r requirements.txt`

### Video Processing Issues
- Check video file format (MP4 recommended)
- Verify video file path is correct
- Adjust HSV color ranges in config for your specific video
