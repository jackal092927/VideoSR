#!/usr/bin/env python3
"""
Toy Car Inclined Plane Analysis Example
========================================
This script demonstrates how to use the enhanced video analysis system
specifically for analyzing a toy car moving down an inclined plane.

The video shows: "a toy car is placed at the top of the inclined plane.
Then it moves down under gravity."

This example configures the system for optimal detection and physics
analysis of this specific scenario.
"""

import sys
import json
from pathlib import Path
from enhanced_analyzer import (
    EnhancedVideoAnalyzer,
    DetectionConfig,
    TraditionalCVConfig,
    DINOConfig,
    LLMConfig,
    PhysicsConfig,
    OutputConfig
)

def create_toy_car_config():
    """Create optimized configuration for toy car analysis"""
    detection_config = DetectionConfig(
        method="hybrid",
        traditional_cv=TraditionalCVConfig(
            enabled=True,
            hsv_lower=[5, 50, 50],      # Adjusted for typical toy car colors (red/orange)
            hsv_upper=[25, 255, 255],
            hsv_lower2=[160, 50, 50],   # Handle red color wrap-around
            hsv_upper2=[179, 255, 255],
            morph_kernel=7,             # Larger kernel for car-shaped objects
            min_area=200,               # Cars are larger than small objects
            roi=None,                   # Let system detect full frame
            invert_threshold=False
        ),
        dino=DINOConfig(
            enabled=True,
            model="facebook/dinov2-base",
            threshold=0.7,              # Higher threshold for car detection
            device="auto",
            segmentation={"enabled": True, "min_mask_area": 1000, "max_mask_area": 50000}
        ),
        llm=LLMConfig(
            enabled=True,
            backend="ollama",  # Use open-source Ollama instead of OpenAI
            model="llava",     # LLaVA vision model for object detection
            system_prompt="""
            You are a physics analysis assistant specializing in vehicle motion on inclined planes.
            Focus on: car detection, trajectory analysis, gravitational effects, friction considerations.
            Provide precise measurements and physics insights for inclined plane scenarios.
            """,
            user_prompt_template="""
            Analyze this frame from a video of a toy car moving down an inclined plane.
            Identify the car and extract: 1) Center coordinates of the car in pixels,
            2) Orientation/pose of the car, 3) Any visible motion blur or speed indicators,
            4) Physics-relevant observations about the inclined plane setup.
            Respond with structured JSON: {"car_center": [x,y], "confidence": 0.0-1.0,
            "orientation": "angle_degrees", "observations": ["obs1", "obs2"]}
            """,
            temperature=0.1,
            max_tokens=300,
            cache_responses=True,
            ollama_host="http://localhost:11434"  # Default Ollama host
        )
    )

    physics_config = PhysicsConfig(
        scenario="inclined_plane",
        gravity=9.81,                  # Standard gravity
        y_axis_down=True,             # Video coordinates: y increases downward
        adaptive_calibration=True,    # Auto-detect scale if possible
        pixels_per_meter=None         # Will be estimated if reference objects visible
    )

    output_config = OutputConfig(
        csv_filename="toy_car_measurements.csv",
        annotated_video_filename="toy_car_annotated.mp4",
        plots_directory="plots",
        include_metadata=True
    )

    return detection_config, physics_config, output_config

def analyze_toy_car_video(video_path: str, output_dir: str = "toy_car_analysis"):
    """
    Analyze a toy car inclined plane video with optimized settings

    Args:
        video_path: Path to the video file
        output_dir: Directory to save analysis results
    """
    print("🎯 Toy Car Inclined Plane Analysis")
    print("=" * 50)
    print(f"Video: {video_path}")
    print(f"Output directory: {output_dir}")
    print()

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    # Create optimized configuration
    print("🔧 Creating optimized configuration for toy car detection...")
    detection_config, physics_config, output_config = create_toy_car_config()

    # Update output paths
    output_config.plots_directory = str(output_path / "plots")
    output_config.csv_filename = "toy_car_measurements.csv"
    output_config.annotated_video_filename = "toy_car_annotated.mp4"

    # Save configuration for reference
    config_summary = {
        "detection_method": detection_config.method,
        "traditional_cv_enabled": detection_config.traditional_cv.enabled,
        "dino_enabled": detection_config.dino.enabled,
        "llm_enabled": detection_config.llm.enabled,
        "llm_backend": detection_config.llm.backend,
        "llm_model": detection_config.llm.model,
        "physics_scenario": physics_config.scenario,
        "gravity": physics_config.gravity
    }

    with open(output_path / "analysis_config.json", 'w') as f:
        json.dump(config_summary, f, indent=2)

    print("✅ Configuration created:")
    print(f"   Detection method: {detection_config.method}")
    print(f"   Traditional CV: {'✓' if detection_config.traditional_cv.enabled else '✗'}")
    print(f"   DINO: {'✓' if detection_config.dino.enabled else '✗'}")
    print(f"   LLM ({detection_config.llm.backend}): {'✓' if detection_config.llm.enabled else '✗'}")
    if detection_config.llm.enabled:
        print(f"      Model: {detection_config.llm.model}")
        print(f"      Backend: {detection_config.llm.backend}")
    print()

    # Scenario description for LLM analysis
    scenario_description = """
    This is a video of a toy car placed at the top of an inclined plane.
    The car moves down the incline under the influence of gravity.
    The inclined plane creates an angle with the horizontal, causing the car
    to accelerate downward. The motion should show characteristics of
    constant acceleration along the incline direction, with possible
    friction effects that might lead to terminal velocity.
    """

    print("🎬 Starting video analysis...")
    print("   This may take a few minutes depending on video length and enabled features")
    print()

    # Create analyzer
    analyzer = EnhancedVideoAnalyzer(detection_config, physics_config, output_config)

    # Run analysis
    try:
        results_df = analyzer.analyze_video(str(video_path), scenario_description)

        # Print results summary
        print("\n" + "=" * 50)
        print("🎉 ANALYSIS COMPLETE!")
        print("=" * 50)

        total_frames = len(results_df)
        successful_detections = results_df['detection_success'].sum()
        detection_rate = successful_detections / total_frames

        print("📊 Results Summary:")
        print(f"   Total frames processed: {total_frames}")
        print(f"   Successful detections: {successful_detections}")
        print(".1f")

        if successful_detections > 0:
            print("\n📈 Motion Analysis:")
            valid_data = results_df.dropna(subset=['x_px', 'y_px'])

            if len(valid_data) > 1:
                # Calculate basic motion stats
                x_start, x_end = valid_data['x_px'].iloc[0], valid_data['x_px'].iloc[-1]
                y_start, y_end = valid_data['y_px'].iloc[0], valid_data['y_px'].iloc[-1]
                total_distance = ((x_end - x_start)**2 + (y_end - y_start)**2)**0.5

                print(f"   Total distance traveled: {total_distance:.1f} pixels")
                print(f"   Start position: ({x_start:.1f}, {y_start:.1f})")
                print(f"   End position: ({x_end:.1f}, {y_end:.1f})")

        print("\n💾 Files saved:")
        print(f"   📄 Measurements CSV: {output_path / output_config.csv_filename}")
        print(f"   🎬 Annotated video: {output_path / output_config.annotated_video_filename}")
        print(f"   📊 Physics analysis: {output_path / 'physics_analysis.json'}")
        print(f"   📈 Plots: {output_path / 'plots/'}")
        print(f"   ⚙️  Configuration: {output_path / 'analysis_config.json'}")

        print("\n🔍 Next steps:")
        print("   1. Review the annotated video to verify detection accuracy")
        print("   2. Check the physics analysis for motion characteristics")
        print("   3. Examine the plots for trajectory and motion patterns")
        print("   4. Adjust configuration if detection needs improvement")

        return results_df

    except Exception as e:
        print(f"\n❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()
        return None

def create_comparison_analysis(video_path: str, output_base: str = "comparison_analysis"):
    """
    Create a comparison analysis using different detection methods
    to evaluate which works best for the toy car scenario.
    """
    print("🔄 Running Comparative Analysis")
    print("=" * 50)

    methods = ["traditional_cv", "dino", "hybrid"]
    results = {}

    for method in methods:
        print(f"\n🎯 Testing method: {method}")

        # Create method-specific config
        detection_config, physics_config, output_config = create_toy_car_config()
        detection_config.method = method

        # Disable other methods
        if method != "traditional_cv":
            detection_config.traditional_cv.enabled = False
        if method != "dino":
            detection_config.dino.enabled = False
        if method != "hybrid":
            detection_config.gpt4v.enabled = False

        # Update output directory
        output_dir = f"{output_base}_{method}"
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        output_config.plots_directory = str(output_path / "plots")

        # Run analysis
        analyzer = EnhancedVideoAnalyzer(detection_config, physics_config, output_config)
        try:
            df = analyzer.analyze_video(video_path, "toy car inclined plane analysis")
            detection_rate = df['detection_success'].mean()
            results[method] = {
                "success_rate": detection_rate,
                "total_frames": len(df),
                "output_dir": output_dir
            }
            print(".1%")
        except Exception as e:
            print(f"❌ Method {method} failed: {e}")
            results[method] = {"error": str(e)}

    # Print comparison summary
    print("\n" + "=" * 50)
    print("📊 METHOD COMPARISON")
    print("=" * 50)

    successful_methods = {k: v for k, v in results.items() if "success_rate" in v}
    if successful_methods:
        best_method = max(successful_methods.items(), key=lambda x: x[1]["success_rate"])
        print(f"🏆 Best performing method: {best_method[0]} ({best_method[1]['success_rate']:.1%})")

        print("\n📋 All results:")
        for method, result in successful_methods.items():
            print("15")

    return results

def main():
    """Main function for command-line usage"""
    import argparse

    parser = argparse.ArgumentParser(description="Toy Car Inclined Plane Analysis")
    parser.add_argument("--video", type=str, required=True,
                       help="Path to the toy car video file")
    parser.add_argument("--output", type=str, default="toy_car_analysis",
                       help="Output directory for results")
    parser.add_argument("--compare", action="store_true",
                       help="Run comparative analysis of different detection methods")
    parser.add_argument("--config", type=str, default=None,
                       help="Path to custom configuration JSON file")

    args = parser.parse_args()

    if not Path(args.video).exists():
        print(f"❌ Video file not found: {args.video}")
        sys.exit(1)

    if args.compare:
        # Run comparative analysis
        results = create_comparison_analysis(args.video, args.output)
    else:
        # Run standard analysis
        if args.config:
            # Load custom config
            print(f"🔧 Loading custom configuration from: {args.config}")
            # Note: In a full implementation, you'd load the config here
            print("⚠️  Custom config loading not implemented in this example")
            print("    Using default toy car configuration instead...")

        results_df = analyze_toy_car_video(args.video, args.output)

        if results_df is not None:
            print(f"\n✅ Analysis completed successfully!")
            print(f"📁 Results saved to: {args.output}")
        else:
            print("\n❌ Analysis failed!")
            sys.exit(1)

if __name__ == "__main__":
    main()
