# Enhanced Video Analysis for Physics Experiments

This repository contains an advanced video analysis system designed for analyzing physics experiments, with special optimizations for scenarios like toy cars moving down inclined planes.

## 🚀 Features

- **Multi-modal Detection**: Combines traditional computer vision, DINOv2 foundation models, and GPT-4V for robust object detection
- **Advanced Physics Analysis**: Automatic scenario detection and physics model fitting
- **Adaptive Configuration**: Self-tuning parameters based on video content
- **Comprehensive Output**: CSV data, annotated videos, physics reports, and visualization plots
- **Modular Design**: Easily extensible for new detection methods and physics scenarios

## 📁 File Structure

```
videoXer/
├── enhanced_prompt.txt           # Enhanced prompt template with advanced tools
├── enhanced_config.json          # Comprehensive configuration file
├── enhanced_analyzer.py          # Main analysis engine
├── toy_car_example.py            # Specialized example for toy car analysis
├── sample_config.json            # Original basic configuration
├── analyze.py                    # Original basic analyzer
├── prompt.txt                    # Original basic prompt
└── README_enhanced_analysis.md   # This documentation
```

## 🛠️ Installation & Setup

### Prerequisites

```bash
pip install opencv-python numpy pandas scipy matplotlib
pip install torch torchvision  # For DINO models
pip install transformers       # For DINO model loading
```

### LLM Backend Setup

Choose one of the following LLM backends:

#### Option 1: OpenAI GPT-4V (Commercial)
```bash
pip install openai
export OPENAI_API_KEY="your-api-key-here"
```

#### Option 2: Ollama (Open-Source, Recommended)
```bash
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Pull vision-capable models
ollama pull llava        # LLaVA 7B model
ollama pull bakllava     # BakLLaVA model
ollama pull moondream    # Moondream model

# Start Ollama service (runs on localhost:11434)
ollama serve
```

The system defaults to using Ollama with LLaVA for open-source vision analysis.

## 🎯 Quick Start - Toy Car Analysis

### Basic Usage

```bash
# Run the toy car analysis example
python toy_car_example.py --video 2419_1744339511.mp4 --output toy_car_results
```

### Advanced Usage

```bash
# Run comparative analysis of different detection methods
python toy_car_example.py --video 2419_1744339511.mp4 --compare --output comparison_results

# Use custom configuration
python toy_car_example.py --video 2419_1744339511.mp4 --config my_custom_config.json
```

### Direct API Usage

```python
from enhanced_analyzer import EnhancedVideoAnalyzer, create_toy_car_config

# Create configuration optimized for toy cars
detection_config, physics_config, output_config = create_toy_car_config()

# Create analyzer
analyzer = EnhancedVideoAnalyzer(detection_config, physics_config, output_config)

# Run analysis
scenario_description = "toy car moving down an inclined plane under gravity"
results_df = analyzer.analyze_video("2419_1744339511.mp4", scenario_description)

print(f"Analysis complete! Processed {len(results_df)} frames")
print(f"Detection success rate: {results_df['detection_success'].mean():.1%}")
```

## ⚙️ Configuration

### Detection Methods

The system supports multiple detection approaches:

1. **Traditional CV** (`traditional_cv`): HSV color thresholding, contour detection
2. **DINO** (`dino`): Facebook's DINOv2 foundation model for object detection
3. **LLM Vision** (`llm`): Multi-backend LLM with vision capabilities
   - **Ollama** (Open-source): LLaVA, BakLLaVA, Moondream models
   - **OpenAI**: GPT-4V (commercial)
4. **Hybrid** (`hybrid`): Weighted fusion of multiple methods

### Physics Scenarios

Automatic detection of:
- `projectile`: Ballistic motion (constant acceleration)
- `inclined_plane`: Motion down an incline (your toy car scenario)
- `pendulum`: Oscillatory motion
- `circular_motion`: Uniform circular motion
- `free_fall`: Vertical motion under gravity
- `collision`: Multi-object interactions

### Configuration Examples

#### Ollama Configuration (Open-Source)
```json
{
  "detection": {
    "method": "hybrid",
    "traditional_cv": {
      "enabled": true,
      "hsv_lower": [5, 50, 50],
      "hsv_upper": [25, 255, 255],
      "min_area": 200
    },
    "dino": {
      "enabled": true,
      "threshold": 0.7,
      "model": "facebook/dinov2-base"
    },
    "llm": {
      "enabled": true,
      "backend": "ollama",
      "model": "llava",
      "ollama_host": "http://localhost:11434",
      "temperature": 0.1,
      "max_tokens": 300
    }
  },
  "physics": {
    "scenario": "inclined_plane",
    "gravity": 9.81,
    "y_axis_down": true
  }
}
```

#### OpenAI Configuration (Commercial)
```json
{
  "detection": {
    "method": "hybrid",
    "llm": {
      "enabled": true,
      "backend": "openai",
      "model": "gpt-4-vision-preview",
      "api_key_env": "OPENAI_API_KEY",
      "temperature": 0.1,
      "max_tokens": 300
    }
  }
}
```

## 📊 Output Files

The analysis generates several output files:

### Data Files
- `enhanced_measurements.csv`: Per-frame measurements with timestamps and coordinates
- `physics_analysis.json`: Detailed physics analysis results
- `analysis_config.json`: Configuration used for the analysis

### Visualization Files
- `plots/trajectory.png`: Object trajectory plot
- `plots/position_time.png`: Position vs time graphs
- `plots/physics_summary.png`: Physics parameters visualization
- `enhanced_annotated.mp4`: Original video with detection overlays

### Physics Analysis Results

The system provides:
- Motion characteristics (speed, acceleration, displacement)
- Physics model fitting (projectile, inclined plane, etc.)
- Fit quality metrics (R-squared values)
- Scenario-specific insights

## 🔧 Advanced Usage

### Custom Detection Methods

Extend the system by implementing new detectors:

```python
from enhanced_analyzer import HybridDetector, DetectionConfig

class MyCustomDetector:
    def detect(self, frame):
        # Your custom detection logic
        return (x, y), confidence

# Add to detection config
config = DetectionConfig()
config.custom_detector = MyCustomDetector()
```

### Physics Model Extensions

Add new physics models:

```python
from enhanced_analyzer import PhysicsAnalyzer

class MyPhysicsModel:
    def fit_trajectory(self, x, y, t):
        # Your physics model fitting
        return fitted_parameters

# Extend the physics analyzer
analyzer = PhysicsAnalyzer(physics_config)
analyzer.add_model("my_model", MyPhysicsModel())
```

### Batch Processing

Process multiple videos:

```python
videos = ["video1.mp4", "video2.mp4", "video3.mp4"]
scenarios = ["projectile", "inclined_plane", "pendulum"]

for video, scenario in zip(videos, scenarios):
    results = analyzer.analyze_video(video, f"Analysis of {scenario} motion")
    print(f"Completed {scenario}: {len(results)} frames")
```

## 🎨 Visualization & Reporting

### Custom Plots

```python
import matplotlib.pyplot as plt
import pandas as pd

# Load results
df = pd.read_csv("enhanced_measurements.csv")

# Create custom visualization
plt.figure(figsize=(12, 8))
valid_data = df.dropna(subset=['x_px', 'y_px'])
plt.plot(valid_data['x_px'], valid_data['y_px'], 'r.-', linewidth=2)
plt.xlabel('X Position (pixels)')
plt.ylabel('Y Position (pixels)')
plt.title('Custom Trajectory Plot')
plt.grid(True)
plt.savefig('custom_trajectory.png', dpi=300)
```

### Physics Report Integration

```python
import json

# Load physics analysis
with open('physics_analysis.json', 'r') as f:
    physics = json.load(f)

# Extract key insights
motion_chars = physics['motion_characteristics']
physics_model = physics['physics_model']

print(f"Average speed: {motion_chars['avg_speed']:.2f} px/s")
print(f"Physics scenario: {physics['detected_scenario']}")
```

## 🐛 Troubleshooting

### Common Issues

1. **Low Detection Rate**
   - Adjust HSV color ranges in config
   - Try different detection methods
   - Check video lighting and contrast

2. **GPT-4V API Errors**
   - Verify OPENAI_API_KEY is set
   - Check API rate limits
   - Consider using cached responses

3. **Memory Issues with DINO**
   - Use smaller DINO model (`dinov2-small`)
   - Process video in batches
   - Reduce patch size

4. **Poor Physics Fitting**
   - Ensure sufficient valid detections (>10 frames)
   - Check coordinate system orientation
   - Verify physics scenario selection

### Debug Mode

Enable detailed logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Run analysis with debug output
results = analyzer.analyze_video(video_path, scenario_description)
```

## 📚 Examples & Tutorials

### Example 1: Projectile Motion

```python
# Configure for ball tracking
detection_config = DetectionConfig(
    method="hybrid",
    traditional_cv=TraditionalCVConfig(
        hsv_lower=[35, 50, 50],  # Yellow ball
        hsv_upper=[45, 255, 255],
        min_area=50
    )
)

physics_config = PhysicsConfig(scenario="projectile")
analyzer = EnhancedVideoAnalyzer(detection_config, physics_config, output_config)
results = analyzer.analyze_video("ball_throw.mp4", "ball thrown in projectile motion")
```

### Example 2: Pendulum Analysis

```python
# Configure for pendulum bob
physics_config = PhysicsConfig(scenario="pendulum")
analyzer = EnhancedVideoAnalyzer(detection_config, physics_config, output_config)
results = analyzer.analyze_video("pendulum.mp4", "simple pendulum oscillation")
```

## 🤝 Contributing

To extend the system:

1. **New Detection Methods**: Implement the detector interface
2. **Physics Models**: Add fitting algorithms to PhysicsAnalyzer
3. **Visualization**: Create new plot types in the output generation
4. **Configuration**: Extend the config dataclasses for new parameters

## 📄 License

This enhanced video analysis system is provided as-is for educational and research purposes.

## 🔄 Migration from Basic Version

If you're upgrading from the basic analyzer:

1. **Configuration**: The enhanced config extends the basic format
2. **API Changes**: Main class renamed from `VideoAnalyzer` to `EnhancedVideoAnalyzer`
3. **New Features**: GPT-4V and DINO support added (optional)
4. **Output Format**: Additional metadata and physics analysis included

### Migration Example

```python
# Old way
from analyze import VideoAnalyzer
analyzer = VideoAnalyzer()
results = analyzer.analyze_video("video.mp4")

# New way
from enhanced_analyzer import EnhancedVideoAnalyzer, create_default_configs
detection_config, physics_config, output_config = create_default_configs()
analyzer = EnhancedVideoAnalyzer(detection_config, physics_config, output_config)
results = analyzer.analyze_video("video.mp4", "physics scenario description")
```

## 📞 Support

For issues and questions:
1. Check the troubleshooting section
2. Review the example configurations
3. Examine the generated log files
4. Test with different detection methods

---

**Happy analyzing! 🎬🔬**
