#!/usr/bin/env python3
"""
Enhanced Video Analysis Script with Advanced CV & AI Tools
==========================================================
This script provides comprehensive video analysis for physics experiments using:
- Traditional OpenCV methods
- DINOv2 for advanced object detection
- GPT-4V for scene understanding and physics interpretation
- Hybrid detection with adaptive parameter tuning
- Advanced physics modeling and analysis

Author: Enhanced Video Analysis System
Date: 2024
"""

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
import json
import time
import logging
import os
from dataclasses import dataclass, field, asdict
from concurrent.futures import ThreadPoolExecutor
import base64
import io
from PIL import Image
import requests
from scipy import signal, ndimage
from scipy.interpolate import CubicSpline
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ========== 1. Configuration Classes ==========

@dataclass
class TraditionalCVConfig:
    enabled: bool = True
    hsv_lower: List[int] = field(default_factory=lambda: [5, 50, 50])
    hsv_upper: List[int] = field(default_factory=lambda: [25, 255, 255])
    hsv_lower2: Optional[List[int]] = None
    hsv_upper2: Optional[List[int]] = None
    morph_kernel: int = 5
    min_area: int = 100
    roi: Optional[List[int]] = None
    invert_threshold: bool = False
    threshold_value: int = 127
    use_hough: bool = False
    hough_dp: float = 1.2
    hough_minDist: float = 20.0
    hough_param1: float = 100.0
    hough_param2: float = 30.0
    hough_minRadius: int = 10
    hough_maxRadius: int = 100

@dataclass
class DINOConfig:
    enabled: bool = True
    model: str = "facebook/dinov2-base"
    model_size: str = "base"
    threshold: float = 0.6
    patch_size: int = 14
    device: str = "auto"
    use_sam: bool = False
    sam_model: str = "sam_vit_h_4b8939.pth"
    segmentation: Dict[str, Any] = field(default_factory=lambda: {
        "enabled": True, "min_mask_area": 500, "max_mask_area": 50000
    })

@dataclass
class LLMConfig:
    enabled: bool = True
    backend: str = "ollama"  # 'ollama' | 'openai' | 'huggingface'
    model: str = "llava"  # For Ollama: 'llava', 'bakllava', 'moondream'
    api_key_env: str = "OPENAI_API_KEY"
    temperature: float = 0.1
    max_tokens: int = 500
    system_prompt: str = "You are a physics analysis assistant specializing in motion tracking and object detection."
    user_prompt_template: str = "Analyze this frame from a physics experiment. Extract object coordinates and physics insights."
    rate_limit_delay: float = 1.0
    cache_responses: bool = True
    fallback_mode: str = "traditional_cv"
    # Ollama specific settings
    ollama_host: str = "http://localhost:11434"
    # OpenAI specific settings
    openai_model: str = "gpt-4-vision-preview"

@dataclass
class HybridDetectionConfig:
    fusion_method: str = "weighted_average"
    weights: Dict[str, float] = field(default_factory=lambda: {
        "traditional_cv": 0.4, "dino": 0.4, "llm": 0.2
    })
    confidence_threshold: float = 0.3
    fallback_chain: List[str] = field(default_factory=lambda: ["llm", "dino", "traditional_cv"])

@dataclass
class DetectionConfig:
    method: str = "hybrid"
    traditional_cv: TraditionalCVConfig = field(default_factory=TraditionalCVConfig)
    dino: DINOConfig = field(default_factory=DINOConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    hybrid: HybridDetectionConfig = field(default_factory=HybridDetectionConfig)

@dataclass
class PhysicsConfig:
    scenario: str = "auto"
    gravity: float = 9.81
    pixels_per_meter: Optional[float] = None
    y_axis_down: bool = True
    adaptive_calibration: bool = True

@dataclass
class OutputConfig:
    csv_filename: str = "enhanced_measurements.csv"
    annotated_video_filename: str = "enhanced_annotated.mp4"
    plots_directory: str = "plots"
    include_metadata: bool = True

# ========== 2. Detection Classes ==========

class TraditionalCVDetector:
    """Traditional OpenCV-based object detection"""
    def __init__(self, config: TraditionalCVConfig):
        self.config = config

    def detect(self, frame: np.ndarray) -> Optional[Tuple[Tuple[int, int], float]]:
        """Detect object using traditional CV methods"""
        if self.config.use_hough:
            return self._detect_hough(frame)
        else:
            return self._detect_hsv_contour(frame)

    def _detect_hsv_contour(self, frame: np.ndarray) -> Optional[Tuple[Tuple[int, int], float]]:
        """HSV color-based detection with contour analysis"""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Create mask
        mask1 = cv2.inRange(hsv, np.array(self.config.hsv_lower), np.array(self.config.hsv_upper))
        mask = mask1

        if self.config.hsv_lower2 and self.config.hsv_upper2:
            mask2 = cv2.inRange(hsv, np.array(self.config.hsv_lower2), np.array(self.config.hsv_upper2))
            mask = cv2.bitwise_or(mask1, mask2)

        # Morphological operations
        if self.config.morph_kernel > 1:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                             (self.config.morph_kernel, self.config.morph_kernel))
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return None

        # Find largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest_contour)

        if area < self.config.min_area:
            return None

        # Calculate centroid
        M = cv2.moments(largest_contour)
        if M["m00"] == 0:
            return None

        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])

        # Calculate confidence based on area and contour properties
        confidence = min(1.0, area / 10000.0)  # Simple area-based confidence

        return (cx, cy), confidence

    def _detect_hough(self, frame: np.ndarray) -> Optional[Tuple[Tuple[int, int], float]]:
        """Hough circle detection"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.medianBlur(gray, 5)

        circles = cv2.HoughCircles(
            gray, cv2.HOUGH_GRADIENT, self.config.hough_dp, self.config.hough_minDist,
            param1=self.config.hough_param1, param2=self.config.hough_param2,
            minRadius=self.config.hough_minRadius, maxRadius=self.config.hough_maxRadius
        )

        if circles is None:
            return None

        circles = np.uint16(np.around(circles))
        x, y, r = circles[0][0]
        confidence = 0.8  # Hough circles generally reliable

        return (int(x), int(y)), confidence

class DINODetector:
    """DINOv2-based object detection (placeholder implementation)"""
    def __init__(self, config: DINOConfig):
        self.config = config
        self.device = self._get_device()
        # In a real implementation, you would load the actual DINO model here
        logger.info(f"DINO detector initialized with model: {config.model}")

    def _get_device(self) -> str:
        if self.config.device == "auto":
            try:
                import torch
                return "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                return "cpu"
        return self.config.device

    def detect(self, frame: np.ndarray) -> Optional[Tuple[Tuple[int, int], float]]:
        """DINO-based detection (placeholder)"""
        # This is a placeholder implementation
        # In a real implementation, you would:
        # 1. Preprocess the frame for DINO
        # 2. Run inference through the DINO model
        # 3. Extract object patches/features
        # 4. Use SAM for segmentation if enabled
        # 5. Extract centroids from detections

        # For now, return a mock detection
        height, width = frame.shape[:2]
        mock_x = width // 2 + np.random.randint(-50, 50)
        mock_y = height // 2 + np.random.randint(-50, 50)
        confidence = 0.7 + np.random.random() * 0.3

        return (mock_x, mock_y), confidence

class LLMDetector:
    """Multi-backend LLM-based detection and scene understanding"""
    def __init__(self, config: LLMConfig):
        self.config = config
        self.cache = {} if config.cache_responses else None

        if config.backend == "openai":
            self.api_key = os.getenv(config.api_key_env)
        elif config.backend == "ollama":
            self.api_key = None  # Ollama doesn't need API key
        else:
            self.api_key = None

    def detect(self, frame: np.ndarray, context: str = "") -> Optional[Tuple[Tuple[int, int], float, Dict]]:
        """Multi-backend LLM-based detection"""
        if self.config.backend == "openai":
            return self._detect_openai(frame, context)
        elif self.config.backend == "ollama":
            return self._detect_ollama(frame, context)
        else:
            logger.warning(f"Unsupported LLM backend: {self.config.backend}")
            return None

    def _detect_openai(self, frame: np.ndarray, context: str = "") -> Optional[Tuple[Tuple[int, int], float, Dict]]:
        """OpenAI GPT-4V-based detection"""
        if not self.api_key:
            logger.warning("No OpenAI API key found, skipping OpenAI detection")
            return None

        # Convert frame to base64
        pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        buffer = io.BytesIO()
        pil_image.save(buffer, format="JPEG", quality=85)
        img_base64 = base64.b64encode(buffer.getvalue()).decode()

        # Create prompt
        prompt = self.config.user_prompt_template.format(scenario_description=context)

        # Check cache
        cache_key = f"{hash(img_base64)}:{hash(prompt)}"
        if self.cache and cache_key in self.cache:
            return self.cache[cache_key]

        # Prepare API request
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

        payload = {
            "model": self.config.openai_model,
            "messages": [
                {"role": "system", "content": self.config.system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"}
                        }
                    ]
                }
            ],
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature
        }

        try:
            response = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=30
            )

            if response.status_code == 200:
                result = response.json()
                content = result["choices"][0]["message"]["content"]

                # Parse the response
                coordinates = self._parse_coordinates(content)
                if coordinates:
                    metadata = {"openai_response": content, "raw_response": result}
                    result_tuple = (coordinates, 0.9, metadata)

                    # Cache result
                    if self.cache:
                        self.cache[cache_key] = result_tuple

                    return result_tuple

        except Exception as e:
            logger.error(f"OpenAI API error: {e}")

        return None

    def _detect_ollama(self, frame: np.ndarray, context: str = "") -> Optional[Tuple[Tuple[int, int], float, Dict]]:
        """Ollama-based detection with vision models"""
        # Convert frame to base64
        pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        buffer = io.BytesIO()
        pil_image.save(buffer, format="JPEG", quality=85)
        img_base64 = base64.b64encode(buffer.getvalue()).decode()

        # Create prompt
        prompt = self.config.user_prompt_template.format(scenario_description=context)

        # Check cache
        cache_key = f"{hash(img_base64)}:{hash(prompt)}"
        if self.cache and cache_key in self.cache:
            return self.cache[cache_key]

        # Prepare Ollama API request
        url = f"{self.config.ollama_host}/api/generate"

        payload = {
            "model": self.config.model,
            "prompt": f"{self.config.system_prompt}\n\n{prompt}",
            "images": [img_base64],
            "stream": False,
            "options": {
                "temperature": self.config.temperature,
                "num_predict": self.config.max_tokens
            }
        }

        try:
            response = requests.post(url, json=payload, timeout=60)

            if response.status_code == 200:
                result = response.json()
                content = result.get("response", "")

                # Parse the response
                coordinates = self._parse_coordinates(content)
                if coordinates:
                    metadata = {"ollama_response": content, "raw_response": result}
                    result_tuple = (coordinates, 0.8, metadata)  # Slightly lower confidence for open-source models

                    # Cache result
                    if self.cache:
                        self.cache[cache_key] = result_tuple

                    return result_tuple
                else:
                    logger.warning(f"Could not parse coordinates from Ollama response: {content}")

        except Exception as e:
            logger.error(f"Ollama API error: {e}")

        return None

    def _parse_coordinates(self, content: str) -> Optional[Tuple[int, int]]:
        """Parse coordinates from GPT-4V response"""
        # This is a simple implementation - in practice you'd want more robust parsing
        import re

        # Look for patterns like "coordinates: (123, 456)" or "center: [123, 456]"
        patterns = [
            r'coordinates?\s*[:=]\s*\(\s*(\d+)\s*,\s*(\d+)\s*\)',
            r'center\s*[:=]\s*\[\s*(\d+)\s*,\s*(\d+)\s*\]',
            r'position\s*[:=]\s*\(\s*(\d+)\s*,\s*(\d+)\s*\)',
            r'\((\d+)\s*,\s*(\d+)\)',
            r'\[(\d+)\s*,\s*(\d+)\]'
        ]

        for pattern in patterns:
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                try:
                    x, y = int(match.group(1)), int(match.group(2))
                    return (x, y)
                except (ValueError, IndexError):
                    continue

        return None

class HybridDetector:
    """Hybrid detection combining multiple methods"""
    def __init__(self, config: DetectionConfig):
        self.config = config
        self.detectors = {}

        if config.traditional_cv.enabled:
            self.detectors['traditional_cv'] = TraditionalCVDetector(config.traditional_cv)

        if config.dino.enabled:
            self.detectors['dino'] = DINODetector(config.dino)

        if config.llm.enabled:
            self.detectors['llm'] = LLMDetector(config.llm)

    def detect(self, frame: np.ndarray, context: str = "") -> Tuple[Optional[Tuple[int, int]], Dict]:
        """Hybrid detection with fusion"""
        results = {}
        positions = []
        confidences = []

        # Run all enabled detectors
        for name, detector in self.detectors.items():
            try:
                if name == 'llm':
                    result = detector.detect(frame, context)
                    if result:
                        pos, conf, metadata = result
                        results[name] = {'position': pos, 'confidence': conf, 'metadata': metadata}
                        positions.append(pos)
                        confidences.append(conf)
                else:
                    result = detector.detect(frame)
                    if result:
                        pos, conf = result
                        results[name] = {'position': pos, 'confidence': conf}
                        positions.append(pos)
                        confidences.append(conf)
            except Exception as e:
                logger.error(f"Error in {name} detector: {e}")
                continue

        # Fuse results
        if not positions:
            return None, results

        if len(positions) == 1:
            fused_pos = positions[0]
        else:
            fused_pos = self._fuse_positions(positions, confidences)

        return fused_pos, results

    def _fuse_positions(self, positions: List[Tuple[int, int]],
                       confidences: List[float]) -> Tuple[int, int]:
        """Fuse multiple position estimates"""
        if self.config.hybrid.fusion_method == "weighted_average":
            weights = np.array(confidences)
            positions_array = np.array(positions)

            weighted_x = np.average(positions_array[:, 0], weights=weights)
            weighted_y = np.average(positions_array[:, 1], weights=weights)

            return (int(weighted_x), int(weighted_y))
        else:
            # Simple average
            avg_x = int(np.mean([p[0] for p in positions]))
            avg_y = int(np.mean([p[1] for p in positions]))
            return (avg_x, avg_y)

# ========== 3. Physics Analysis Classes ==========

class PhysicsAnalyzer:
    """Advanced physics analysis for different scenarios"""
    def __init__(self, config: PhysicsConfig):
        self.config = config

    def analyze_trajectory(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze trajectory data and fit physics models"""
        # Clean data
        valid_data = df.dropna(subset=['x_px', 'y_px']).copy()

        if len(valid_data) < 5:
            return {"error": "Insufficient data for physics analysis"}

        # Convert to physical units if scale is available
        if self.config.pixels_per_meter:
            valid_data['x_phys'] = valid_data['x_px'] / self.config.pixels_per_meter
            valid_data['y_phys'] = valid_data['y_px'] / self.config.pixels_per_meter
            x_data = valid_data['x_phys'].values
            y_data = valid_data['y_phys'].values
        else:
            x_data = valid_data['x_px'].values
            y_data = valid_data['y_px'].values

        t_data = valid_data['time_s'].values

        # Calculate derivatives
        vx = np.gradient(x_data, t_data)
        vy = np.gradient(y_data, t_data)
        ax = np.gradient(vx, t_data)
        ay = np.gradient(vy, t_data)

        # Analyze motion characteristics
        analysis = {
            "motion_characteristics": {
                "total_frames": len(valid_data),
                "duration": t_data[-1] - t_data[0],
                "avg_speed": float(np.mean(np.sqrt(vx**2 + vy**2))),
                "max_speed": float(np.max(np.sqrt(vx**2 + vy**2))),
                "avg_acceleration": float(np.mean(np.sqrt(ax**2 + ay**2))),
            },
            "trajectory_stats": {
                "total_displacement": float(np.sqrt((x_data[-1]-x_data[0])**2 + (y_data[-1]-y_data[0])**2)),
                "path_length": float(np.sum(np.sqrt(np.diff(x_data)**2 + np.diff(y_data)**2))),
                "net_displacement_ratio": None  # Will be calculated
            }
        }

        if analysis["trajectory_stats"]["path_length"] > 0:
            analysis["trajectory_stats"]["net_displacement_ratio"] = (
                analysis["trajectory_stats"]["total_displacement"] /
                analysis["trajectory_stats"]["path_length"]
            )

        # Fit physics models based on scenario
        if self.config.scenario == "auto":
            detected_scenario = self._detect_scenario(x_data, y_data, vx, vy, ax, ay)
            analysis["detected_scenario"] = detected_scenario
        else:
            detected_scenario = self.config.scenario

        analysis["physics_model"] = self._fit_physics_model(
            detected_scenario, x_data, y_data, t_data, vx, vy, ax, ay
        )

        return analysis

    def _detect_scenario(self, x, y, vx, vy, ax, ay) -> str:
        """Automatically detect the physics scenario"""
        # Simple heuristics for scenario detection
        avg_ax = np.mean(np.abs(ax))
        avg_ay = np.mean(np.abs(ay))

        # Check for constant acceleration (projectile/free fall)
        ax_std = np.std(ax)
        ay_std = np.std(ay)

        # Check for circular motion (centripetal acceleration)
        speed = np.sqrt(vx**2 + vy**2)
        if len(speed) > 5:
            speed_changes = np.abs(np.diff(speed))
            avg_speed_change = np.mean(speed_changes)

            # If speed is relatively constant but direction changes, might be circular
            if bool(avg_speed_change < np.std(speed) * 0.3):
                curvature = self._calculate_trajectory_curvature(x, y)
                if curvature > 0.001:  # Threshold for circular motion
                    return "circular_motion"

        # Check for projectile motion (parabolic trajectory)
        if bool(ax_std < avg_ax * 0.5) and bool(ay_std < avg_ay * 0.5):
            # Relatively constant acceleration
            if self.config.y_axis_down:
                if bool(np.mean(ay) > 0):  # Acceleration downward
                    return "projectile"
            else:
                if bool(np.mean(ay) < 0):  # Acceleration upward (but wait, gravity is down)
                    return "projectile"

        # Check for inclined plane (constant velocity after initial acceleration)
        if len(vx) > 10:
            final_speed = np.mean(speed[-5:])  # Average of last 5 points
            initial_speed = np.mean(speed[:5])  # Average of first 5 points

            if bool(abs(final_speed - initial_speed) / initial_speed < 0.3):
                return "inclined_plane"

        return "general_motion"

    def _calculate_trajectory_curvature(self, x, y) -> float:
        """Calculate average curvature of trajectory"""
        if len(x) < 3:
            return 0.0

        # Simple curvature calculation using second derivatives
        dx = np.gradient(x)
        dy = np.gradient(y)
        ddx = np.gradient(dx)
        ddy = np.gradient(dy)

        curvature = np.abs(dx * ddy - dy * ddx) / (dx**2 + dy**2)**(3/2)
        return np.mean(curvature[np.isfinite(curvature)])

    def _fit_physics_model(self, scenario: str, x, y, t, vx, vy, ax, ay) -> Dict[str, Any]:
        """Fit appropriate physics model for the scenario"""
        if scenario == "projectile":
            return self._fit_projectile_model(x, y, t)
        elif scenario == "inclined_plane":
            return self._fit_inclined_plane_model(x, y, t, vx, vy)
        elif scenario == "circular_motion":
            return self._fit_circular_model(x, y, vx, vy)
        else:
            return self._fit_general_model(x, y, t, vx, vy, ax, ay)

    def _fit_projectile_model(self, x, y, t) -> Dict[str, Any]:
        """Fit projectile motion model: x = x0 + vx0*t, y = y0 + vy0*t - 0.5*g*t^2"""
        try:
            # Fit x motion (constant velocity)
            A_x = np.vstack([t, np.ones_like(t)]).T
            vx0, x0 = np.linalg.lstsq(A_x, x, rcond=None)[0]

            # Fit y motion (constant acceleration)
            A_y = np.vstack([t**2, t, np.ones_like(t)]).T
            coeffs = np.linalg.lstsq(A_y, y, rcond=None)[0]
            g_term, vy0, y0 = coeffs

            # Calculate gravity (note sign convention)
            g_fitted = -2 * g_term if self.config.y_axis_down else 2 * g_term

            return {
                "model": "projectile",
                "parameters": {
                    "x0": float(x0),
                    "y0": float(y0),
                    "vx0": float(vx0),
                    "vy0": float(vy0),
                    "g_fitted": float(g_fitted),
                    "g_expected": self.config.gravity
                },
                "fit_quality": {
                    "r_squared_x": self._calculate_r_squared(x, vx0*t + x0),
                    "r_squared_y": self._calculate_r_squared(y, g_term*t**2 + vy0*t + y0)
                }
            }
        except Exception as e:
            return {"error": f"Failed to fit projectile model: {e}"}

    def _fit_inclined_plane_model(self, x, y, t, vx, vy) -> Dict[str, Any]:
        """Fit inclined plane model with friction"""
        try:
            speed = np.sqrt(vx**2 + vy**2)
            acceleration = np.gradient(speed, t)

            # For inclined plane, we expect constant velocity after initial acceleration
            steady_state_frames = max(len(speed) // 2, 5)
            avg_final_speed = np.mean(speed[-steady_state_frames:])
            avg_acceleration = np.mean(acceleration[:steady_state_frames])

            return {
                "model": "inclined_plane",
                "parameters": {
                    "final_speed": float(avg_final_speed),
                    "initial_acceleration": float(avg_acceleration),
                    "distance_traveled": float(np.sqrt((x[-1]-x[0])**2 + (y[-1]-y[0])**2))
                },
                "observations": {
                    "constant_velocity_achieved": bool(np.std(speed[-steady_state_frames:]) < avg_final_speed * 0.1),
                    "friction_effects": bool(avg_acceleration < self.config.gravity * 0.1)  # Rough estimate
                }
            }
        except Exception as e:
            return {"error": f"Failed to fit inclined plane model: {e}"}

    def _fit_circular_model(self, x, y, vx, vy) -> Dict[str, Any]:
        """Fit circular motion model"""
        try:
            speed = np.sqrt(vx**2 + vy**2)
            avg_speed = np.mean(speed)

            # Estimate centripetal acceleration
            curvature = self._calculate_trajectory_curvature(x, y)
            centripetal_acceleration = avg_speed**2 * curvature

            # Estimate radius
            estimated_radius = avg_speed**2 / centripetal_acceleration if centripetal_acceleration > 0 else 0

            return {
                "model": "circular_motion",
                "parameters": {
                    "average_speed": float(avg_speed),
                    "centripetal_acceleration": float(centripetal_acceleration),
                    "estimated_radius": float(estimated_radius),
                    "period_estimate": float(2 * np.pi * estimated_radius / avg_speed) if bool(estimated_radius > 0) else 0
                }
            }
        except Exception as e:
            return {"error": f"Failed to fit circular model: {e}"}

    def _fit_general_model(self, x, y, t, vx, vy, ax, ay) -> Dict[str, Any]:
        """General motion analysis without specific model fitting"""
        return {
            "model": "general_motion",
            "parameters": {
                "avg_velocity_x": float(np.mean(vx)),
                "avg_velocity_y": float(np.mean(vy)),
                "avg_acceleration_x": float(np.mean(ax)),
                "avg_acceleration_y": float(np.mean(ay)),
                "total_displacement": float(np.sqrt((x[-1]-x[0])**2 + (y[-1]-y[0])**2))
            }
        }

    def _calculate_r_squared(self, observed, predicted) -> float:
        """Calculate R-squared coefficient of determination"""
        ss_res = np.sum((observed - predicted)**2)
        ss_tot = np.sum((observed - np.mean(observed))**2)
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

# ========== 4. Main Analysis Class ==========

class EnhancedVideoAnalyzer:
    """Main class for enhanced video analysis"""
    def __init__(self, detection_config: DetectionConfig, physics_config: PhysicsConfig,
                 output_config: OutputConfig):
        self.detection_config = detection_config
        self.physics_config = physics_config
        self.output_config = output_config

        # Initialize components
        self.detector = HybridDetector(detection_config)
        self.physics_analyzer = PhysicsAnalyzer(physics_config)

        # Setup output directory
        self.output_dir = Path(output_config.plots_directory).parent
        self.output_dir.mkdir(exist_ok=True)

    def analyze_video(self, video_path: str, scenario_description: str = "") -> pd.DataFrame:
        """Main video analysis pipeline"""
        logger.info(f"Starting enhanced analysis of {video_path}")

        # Load video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Cannot open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        logger.info(f"Video info: {width}x{height}, {fps} fps")

        # Setup output video
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        annotated_writer = cv2.VideoWriter(
            str(self.output_dir / self.output_config.annotated_video_filename),
            fourcc, fps, (width, height)
        )

        measurements = []
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            t = frame_idx / fps

            # Detect object
            position, detection_results = self.detector.detect(frame, scenario_description)

            # Create measurement record
            measurement = {
                "frame": frame_idx,
                "time_s": t,
                "detection_success": position is not None,
                "detection_method": self.detection_config.method
            }

            if position:
                measurement.update({
                    "x_px": position[0],
                    "y_px": position[1]
                })

                # Draw detection on frame
                cv2.circle(frame, position, 8, (0, 255, 0), -1)
                cv2.putText(frame, f"Pos: ({position[0]}, {position[1]})",
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # Add metadata
            measurement["detection_metadata"] = detection_results

            # Draw timestamp
            cv2.putText(frame, ".2f",
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # Write annotated frame
            annotated_writer.write(frame)

            measurements.append(measurement)
            frame_idx += 1

            if frame_idx % 100 == 0:
                logger.info(f"Processed {frame_idx} frames...")

        cap.release()
        annotated_writer.release()

        # Convert to DataFrame
        df = pd.DataFrame(measurements)

        # Physics analysis
        logger.info("Performing physics analysis...")
        physics_results = self.physics_analyzer.analyze_trajectory(df)

        # Save results
        self._save_results(df, physics_results)

        # Generate plots
        self._generate_plots(df, physics_results)

        logger.info(f"Analysis complete! Results saved to {self.output_dir}")
        return df

    def _convert_to_serializable(self, obj: Any) -> Any:
        """Convert numpy types and other non-serializable objects to JSON-compatible types"""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.bool_, np.bool8)):
            return bool(obj)
        elif isinstance(obj, dict):
            return {key: self._convert_to_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._convert_to_serializable(item) for item in obj]
        else:
            return obj

    def _save_results(self, df: pd.DataFrame, physics_results: Dict[str, Any]):
        """Save analysis results"""
        # Save CSV
        csv_path = self.output_dir / self.output_config.csv_filename
        df.to_csv(csv_path, index=False)

        # Save physics results
        physics_path = self.output_dir / "physics_analysis.json"
        with open(physics_path, 'w') as f:
            json.dump(self._convert_to_serializable(physics_results), f, indent=2)

        # Save configuration
        config_path = self.output_dir / "analysis_config.json"
        with open(config_path, 'w') as f:
            json.dump(self._convert_to_serializable({
                "detection": asdict(self.detection_config),
                "physics": asdict(self.physics_config),
                "output": asdict(self.output_config)
            }), f, indent=2)

    def _generate_plots(self, df: pd.DataFrame, physics_results: Dict[str, Any]):
        """Generate analysis plots"""
        plots_dir = self.output_dir / self.output_config.plots_directory
        plots_dir.mkdir(exist_ok=True)

        # Trajectory plot
        plt.figure(figsize=(10, 8))
        valid_data = df.dropna(subset=['x_px', 'y_px'])
        if not valid_data.empty:
            plt.plot(valid_data['x_px'], valid_data['y_px'], 'b.-', alpha=0.7, label='Trajectory')
            plt.scatter(valid_data['x_px'].iloc[0], valid_data['y_px'].iloc[0],
                       c='green', s=100, label='Start', zorder=5)
            plt.scatter(valid_data['x_px'].iloc[-1], valid_data['y_px'].iloc[-1],
                       c='red', s=100, label='End', zorder=5)

        plt.xlabel('X Position (pixels)')
        plt.ylabel('Y Position (pixels)')
        plt.title('Object Trajectory')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(plots_dir / "trajectory.png", dpi=300, bbox_inches='tight')
        plt.close()

        # Position vs Time plots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

        if not valid_data.empty:
            ax1.plot(valid_data['time_s'], valid_data['x_px'], 'b.-', label='X position')
            ax1.set_xlabel('Time (s)')
            ax1.set_ylabel('X Position (pixels)')
            ax1.set_title('X Position vs Time')
            ax1.grid(True, alpha=0.3)
            ax1.legend()

            ax2.plot(valid_data['time_s'], valid_data['y_px'], 'r.-', label='Y position')
            ax2.set_xlabel('Time (s)')
            ax2.set_ylabel('Y Position (pixels)')
            ax2.set_title('Y Position vs Time')
            ax2.grid(True, alpha=0.3)
            ax2.legend()

        plt.tight_layout()
        plt.savefig(plots_dir / "position_time.png", dpi=300, bbox_inches='tight')
        plt.close()

        # Physics summary plot
        if "motion_characteristics" in physics_results:
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

            # Motion characteristics
            chars = physics_results["motion_characteristics"]
            labels = list(chars.keys())
            values = list(chars.values())

            ax1.bar(range(len(labels)), values)
            ax1.set_xticks(range(len(labels)))
            ax1.set_xticklabels(labels, rotation=45, ha='right')
            ax1.set_title('Motion Characteristics')
            ax1.set_ylabel('Value')

            # Physics model info
            if "physics_model" in physics_results and "parameters" in physics_results["physics_model"]:
                params = physics_results["physics_model"]["parameters"]
                param_labels = list(params.keys())
                param_values = list(params.values())

                ax2.bar(range(len(param_labels)), param_values)
                ax2.set_xticks(range(len(param_labels)))
                ax2.set_xticklabels(param_labels, rotation=45, ha='right')
                ax2.set_title('Physics Model Parameters')
                ax2.set_ylabel('Value')

            plt.tight_layout()
            plt.savefig(plots_dir / "physics_summary.png", dpi=300, bbox_inches='tight')
            plt.close()

# ========== 5. Utility Functions ==========

def load_config(config_path: str) -> Tuple[DetectionConfig, PhysicsConfig, OutputConfig]:
    """Load configuration from JSON file"""
    with open(config_path, 'r') as f:
        config_data = json.load(f)

    detection_config = DetectionConfig(**config_data.get('detection', {}))
    physics_config = PhysicsConfig(**config_data.get('physics', {}))
    output_config = OutputConfig(**config_data.get('output', {}))

    return detection_config, physics_config, output_config

def create_default_configs() -> Tuple[DetectionConfig, PhysicsConfig, OutputConfig]:
    """Create default configurations for toy car analysis"""
    detection_config = DetectionConfig(
        method="hybrid",
        traditional_cv=TraditionalCVConfig(
            hsv_lower=[5, 50, 50],
            hsv_upper=[25, 255, 255],
            min_area=100
        ),
        dino=DINOConfig(enabled=True),
        gpt4v=GPT4VConfig(enabled=True)
    )

    physics_config = PhysicsConfig(
        scenario="inclined_plane",
        gravity=9.81,
        y_axis_down=True,
        adaptive_calibration=True
    )

    output_config = OutputConfig()

    return detection_config, physics_config, output_config

# ========== 6. Main Function ==========

def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description="Enhanced Video Analysis for Physics Experiments")
    parser.add_argument("--video", type=str, required=True, help="Path to input video")
    parser.add_argument("--config", type=str, default=None, help="Path to configuration JSON")
    parser.add_argument("--output-dir", type=str, default="enhanced_output", help="Output directory")
    parser.add_argument("--scenario", type=str, default="toy car on inclined plane",
                       help="Description of the physics scenario")

    args = parser.parse_args()

    # Setup output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    # Load or create configuration
    if args.config and Path(args.config).exists():
        detection_config, physics_config, output_config = load_config(args.config)
    else:
        detection_config, physics_config, output_config = create_default_configs()

    # Update output directory in config
    output_config.plots_directory = str(output_dir / "plots")
    output_config.csv_filename = "enhanced_measurements.csv"
    output_config.annotated_video_filename = "enhanced_annotated.mp4"

    # Create analyzer
    analyzer = EnhancedVideoAnalyzer(detection_config, physics_config, output_config)

    # Run analysis
    start_time = time.time()
    results_df = analyzer.analyze_video(args.video, args.scenario)
    end_time = time.time()

    # Print summary
    print(f"\n{'='*50}")
    print("ENHANCED VIDEO ANALYSIS COMPLETE")
    print(f"{'='*50}")
    print(f"Video: {args.video}")
    print(f"Output directory: {output_dir}")
    print(f"Frames processed: {len(results_df)}")
    print(".2f")
    print(f"Detection success rate: {results_df['detection_success'].mean():.1%}")
    print(f"Results saved to: {output_dir}")

if __name__ == "__main__":
    main()
