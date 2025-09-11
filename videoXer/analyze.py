
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
analyze.py — Automatic Video Analysis & Physics Fitting
-------------------------------------------------------
Pipeline:
  (1) Load JSON config & video
  (2) Detect / track object
  (3) Extract per-frame measurements
  (4) Optional smoothing & derivatives
  (5) Optional physics fitting (e.g., projectile)
  (6) Save outputs (CSV + optional plots/annotated video)

Requires: opencv-python, numpy, pandas, scipy, matplotlib
"""

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple, List, Dict

import cv2
import numpy as np
import pandas as pd

# Optional: smoothing / fitting
try:
    from scipy.signal import savgol_filter
except Exception:
    savgol_filter = None

import matplotlib
matplotlib.use("Agg")  # for headless environments
import matplotlib.pyplot as plt


# ---------------------------
# Dataclasses for configuration
# ---------------------------

@dataclass
class DetectionConfig:
    method: str = "hsv"  # 'hsv' | 'hough' | 'threshold'
    # HSV ranges for color thresholding (e.g., red ball -> use two ranges)
    hsv_lower: Optional[List[int]] = None  # [H,S,V]
    hsv_upper: Optional[List[int]] = None
    hsv_lower2: Optional[List[int]] = None  # for red wrap-around
    hsv_upper2: Optional[List[int]] = None
    morph_kernel: int = 5
    min_area: int = 50
    use_hough_grad: bool = False  # for 'hough': use gradient
    hough_dp: float = 1.2
    hough_minDist: float = 20.0
    hough_param1: float = 100.0
    hough_param2: float = 30.0
    hough_minRadius: int = 3
    hough_maxRadius: int = 0
    roi: Optional[List[int]] = None  # [x, y, w, h]
    invert_threshold: bool = False
    threshold_value: int = 127

@dataclass
class TrackingConfig:
    interpolate_max_gap: int = 5   # max consecutive NaNs to interpolate across
    kalman: bool = False           # future extension

@dataclass
class SmoothingConfig:
    method: str = "savgol"         # 'none' | 'savgol' | 'gauss1d'
    window: int = 9
    polyorder: int = 2
    sigma: float = 1.0

@dataclass
class PhysicsConfig:
    fit: str = "projectile"        # 'none' | 'projectile' | 'linear' | 'freefall' | 'harmonic'
    gravity: float = 9.81          # m/s^2 if scale provided
    pixels_per_meter: Optional[float] = None  # if provided, convert px->m
    y_axis_down: bool = True       # True if image y increases downward

@dataclass
class OutputConfig:
    annotated_video: bool = True
    plots: bool = True
    draw_trail: int = 50           # number of past points to draw
    csv_name: str = "measurements.csv"
    annotated_name: str = "annotated.mp4"
    plots_prefix: str = "plot"

@dataclass
class AppConfig:
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    smoothing: SmoothingConfig = field(default_factory=SmoothingConfig)
    physics: PhysicsConfig = field(default_factory=PhysicsConfig)
    output: OutputConfig = field(default_factory=OutputConfig)


# ---------------------------
# Utility functions
# ---------------------------

def load_config(path: Path) -> AppConfig:
    with open(path, "r") as f:
        raw = json.load(f)

    def dataclass_from_dict(dc, d):
        return dc(**{k: d.get(k, getattr(dc(), k)) for k in dc().__dict__.keys()})

    return AppConfig(
        detection=dataclass_from_dict(DetectionConfig, raw.get("detection", {})),
        tracking=dataclass_from_dict(TrackingConfig, raw.get("tracking", {})),
        smoothing=dataclass_from_dict(SmoothingConfig, raw.get("smoothing", {})),
        physics=dataclass_from_dict(PhysicsConfig, raw.get("physics", {})),
        output=dataclass_from_dict(OutputConfig, raw.get("output", {})),
    )


def load_video(video_path: Path):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    return cap, float(fps), (width, height)


def apply_roi(frame: np.ndarray, roi: Optional[List[int]]) -> Tuple[np.ndarray, Tuple[int, int]]:
    """Crop to ROI if provided; return cropped frame and top-left offset (ox, oy)."""
    if roi is None:
        return frame, (0, 0)
    x, y, w, h = roi
    h_img, w_img = frame.shape[:2]
    x0 = max(0, min(x, w_img-1))
    y0 = max(0, min(y, h_img-1))
    x1 = max(1, min(x0 + w, w_img))
    y1 = max(1, min(y0 + h, h_img))
    cropped = frame[y0:y1, x0:x1]
    return cropped, (x0, y0)


def detect_hsv(frame_bgr: np.ndarray, cfg: DetectionConfig) -> Optional[Tuple[int, int]]:
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    mask = None
    if cfg.hsv_lower is not None and cfg.hsv_upper is not None:
        m1 = cv2.inRange(hsv, np.array(cfg.hsv_lower, dtype=np.uint8),
                               np.array(cfg.hsv_upper, dtype=np.uint8))
        mask = m1 if mask is None else (mask | m1)
    if cfg.hsv_lower2 is not None and cfg.hsv_upper2 is not None:
        m2 = cv2.inRange(hsv, np.array(cfg.hsv_lower2, dtype=np.uint8),
                               np.array(cfg.hsv_upper2, dtype=np.uint8))
        mask = m2 if mask is None else (mask | m2)
    if mask is None:
        # fallback: entire frame
        mask = np.ones(hsv.shape[:2], dtype=np.uint8) * 255

    if cfg.morph_kernel and cfg.morph_kernel > 1:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (cfg.morph_kernel, cfg.morph_kernel))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)

    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    c = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(c) < cfg.min_area:
        return None
    M = cv2.moments(c)
    if M["m00"] <= 0:
        return None
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    return cx, cy


def detect_threshold(frame_bgr: np.ndarray, cfg: DetectionConfig) -> Optional[Tuple[int, int]]:
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    thresh_type = cv2.THRESH_BINARY_INV if cfg.invert_threshold else cv2.THRESH_BINARY
    _, mask = cv2.threshold(gray, cfg.threshold_value, 255, thresh_type)

    if cfg.morph_kernel and cfg.morph_kernel > 1:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (cfg.morph_kernel, cfg.morph_kernel))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)

    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    c = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(c) < cfg.min_area:
        return None
    M = cv2.moments(c)
    if M["m00"] <= 0:
        return None
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    return cx, cy


def detect_hough(frame_bgr: np.ndarray, cfg: DetectionConfig) -> Optional[Tuple[int, int]]:
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)
    circles = cv2.HoughCircles(
        gray, cv2.HOUGH_GRADIENT if cfg.use_hough_grad else cv2.HOUGH_GRADIENT_ALT,
        dp=cfg.hough_dp, minDist=cfg.hough_minDist,
        param1=cfg.hough_param1, param2=cfg.hough_param2,
        minRadius=cfg.hough_minRadius, maxRadius=cfg.hough_maxRadius
    )
    if circles is None:
        return None
    circles = np.uint16(np.around(circles))
    # pick largest (or first)
    x, y, r = circles[0][0]
    return int(x), int(y)


def detector_dispatch(frame_bgr: np.ndarray, cfg: DetectionConfig) -> Optional[Tuple[int, int]]:
    if cfg.method.lower() == "hsv":
        return detect_hsv(frame_bgr, cfg)
    elif cfg.method.lower() == "hough":
        return detect_hough(frame_bgr, cfg)
    elif cfg.method.lower() == "threshold":
        return detect_threshold(frame_bgr, cfg)
    else:
        # default fallback: threshold
        return detect_threshold(frame_bgr, cfg)


def interpolate_positions(df: pd.DataFrame, max_gap: int) -> pd.DataFrame:
    # Linear interpolation for short gaps
    df_interp = df.copy()
    df_interp[["x_px", "y_px"]] = df_interp[["x_px", "y_px"]].astype(float)

    # Only interpolate gaps that are <= max_gap
    for col in ["x_px", "y_px"]:
        is_nan = df_interp[col].isna()
        if not is_nan.any():
            continue
        # identify consecutive NaN runs
        runs = []
        start = None
        for i, nan in enumerate(is_nan):
            if nan and start is None:
                start = i
            if not nan and start is not None:
                runs.append((start, i-1))
                start = None
        if start is not None:
            runs.append((start, len(is_nan)-1))

        for s, e in runs:
            run_len = e - s + 1
            if run_len <= max_gap:
                df_interp.loc[s:e, col] = np.interp(
                    np.arange(s, e+1),
                    np.arange(len(df_interp))[~is_nan],
                    df_interp[col][~is_nan]
                )
    return df_interp


def smooth_series(series: np.ndarray, cfg: SmoothingConfig) -> np.ndarray:
    if cfg.method.lower() == "none":
        return series.copy()
    if cfg.method.lower() == "savgol":
        if savgol_filter is None:
            return series.copy()
        # window must be odd and <= len(series)
        w = max(3, cfg.window if cfg.window % 2 == 1 else cfg.window + 1)
        w = min(w, len(series) - (1 - len(series) % 2)) if len(series) > 2 else 3
        w = max(3, w)
        poly = min(cfg.polyorder, w - 1)
        try:
            return savgol_filter(series, window_length=w, polyorder=poly, mode="interp")
        except Exception:
            return series.copy()
    if cfg.method.lower() == "gauss1d":
        # simple gaussian smoothing via convolution
        radius = max(1, int(3*cfg.sigma))
        x = np.arange(-radius, radius+1)
        k = np.exp(-(x**2) / (2*cfg.sigma*cfg.sigma))
        k /= k.sum()
        return np.convolve(series, k, mode="same")
    return series.copy()


def compute_derivatives(x: np.ndarray, y: np.ndarray, t: np.ndarray) -> Dict[str, np.ndarray]:
    # central differences via gradient
    vx = np.gradient(x, t, edge_order=2)
    vy = np.gradient(y, t, edge_order=2)
    ax = np.gradient(vx, t, edge_order=2)
    ay = np.gradient(vy, t, edge_order=2)
    speed = np.sqrt(vx**2 + vy**2)
    accel = np.sqrt(ax**2 + ay**2)
    return {"vx": vx, "vy": vy, "ax": ax, "ay": ay, "speed": speed, "accel": accel}


def fit_projectile(t: np.ndarray, x: np.ndarray, y: np.ndarray, y_axis_down: bool):
    """Fit x = d*t + e; y = a*t^2 + b*t + c. Return params & predicted curves."""
    # Fit x
    A_x = np.vstack([t, np.ones_like(t)]).T
    d, e = np.linalg.lstsq(A_x, x, rcond=None)[0]

    # Fit y
    A_y = np.vstack([t**2, t, np.ones_like(t)]).T
    a, b, c = np.linalg.lstsq(A_y, y, rcond=None)[0]

    # Physical interpretation
    # y'' = 2a (in pixel units). If y increases downward, then 2a ~ +g_px.
    g_px = 2 * a
    if not y_axis_down:
        # if y increases upward, gravity should be negative (downward)
        g_px = -2 * a

    return {"a": a, "b": b, "c": c, "d": d, "e": e, "g_px": g_px,
            "x_fit": d * t + e, "y_fit": a * t**2 + b * t + c}


def maybe_scale_to_meters(px_values: np.ndarray, pixels_per_meter: Optional[float]) -> np.ndarray:
    if pixels_per_meter and pixels_per_meter > 0:
        return px_values / pixels_per_meter
    return px_values


def draw_annotations(frame: np.ndarray, pt: Optional[Tuple[int,int]], trail: List[Tuple[int,int]],
                     text_lines: List[str]):
    # draw path
    if len(trail) >= 2:
        for i in range(1, len(trail)):
            cv2.line(frame, trail[i-1], trail[i], (0, 255, 0), 2)
    # draw point
    if pt is not None:
        cv2.circle(frame, pt, 6, (0, 0, 255), -1)
    # draw text
    y0 = 20
    for line in text_lines:
        cv2.putText(frame, line, (10, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
        y0 += 18


def main():
    parser = argparse.ArgumentParser(description="Automatic video analysis & physics fitting")
    parser.add_argument("--config", type=str, required=False, default=None, help="Path to JSON config")
    parser.add_argument("--video", type=str, required=False, default="/mnt/data/2419_1744339511.mp4", help="Path to input video")
    parser.add_argument("--outdir", type=str, required=False, default="analysis_out", help="Output directory")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Load configuration
    if args.config:
        cfg = load_config(Path(args.config))
    else:
        # Reasonable defaults for a red ball (projectile) style video
        cfg = AppConfig(
            detection=DetectionConfig(
                method="hsv",
                hsv_lower=[0, 90, 70], hsv_upper=[10, 255, 255],
                hsv_lower2=[160, 90, 70], hsv_upper2=[179, 255, 255],
                morph_kernel=5, min_area=40
            ),
            tracking=TrackingConfig(interpolate_max_gap=5, kalman=False),
            smoothing=SmoothingConfig(method="savgol", window=9, polyorder=2, sigma=1.0),
            physics=PhysicsConfig(fit="projectile", gravity=9.81, pixels_per_meter=None, y_axis_down=True),
            output=OutputConfig(annotated_video=True, plots=True, draw_trail=50,
                                csv_name="measurements.csv", annotated_name="annotated.mp4", plots_prefix="plot")
        )

    # Load video
    cap, fps, (W, H) = load_video(Path(args.video))

    # Prepare annotated writer
    writer = None
    if cfg.output.annotated_video:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(str(outdir / cfg.output.annotated_name), fourcc, fps, (W, H))

    # Extraction loop
    records = []
    trail = []
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        t = frame_idx / fps

        # ROI
        view, (ox, oy) = apply_roi(frame, cfg.detection.roi)

        # detection
        pos = detector_dispatch(view, cfg.detection)
        if pos is not None:
            cx, cy = pos
            # map back to full-frame coordinates
            cx_full, cy_full = cx + ox, cy + oy
            records.append({"frame": frame_idx, "time_s": t, "x_px": cx_full, "y_px": cy_full})
            pt_draw = (int(cx_full), int(cy_full))
            trail.append(pt_draw)
            if len(trail) > cfg.output.draw_trail:
                trail = trail[-cfg.output.draw_trail:]
        else:
            records.append({"frame": frame_idx, "time_s": t, "x_px": np.nan, "y_px": np.nan})
            pt_draw = None

        # annotate & write
        if writer is not None:
            text_lines = [f"frame={frame_idx}", f"t={t:.3f}s"]
            draw_annotations(frame, pt_draw, trail, text_lines)
            writer.write(frame)

        frame_idx += 1

    cap.release()
    if writer is not None:
        writer.release()

    df = pd.DataFrame.from_records(records).sort_values("frame").reset_index(drop=True)

    # Interpolate small gaps
    df = interpolate_positions(df, cfg.tracking.interpolate_max_gap)

    # Smooth & derivatives
    x_px = df["x_px"].to_numpy()
    y_px = df["y_px"].to_numpy()
    t = df["time_s"].to_numpy()

    x_smooth = smooth_series(x_px, cfg.smoothing)
    y_smooth = smooth_series(y_px, cfg.smoothing)

    # If y-axis is downward in image coords, keep as-is. If upward, invert for "physics-y"
    if not cfg.physics.y_axis_down:
        y_smooth = -y_smooth

    # Scale to meters if requested (positions)
    x_phys = maybe_scale_to_meters(x_smooth, cfg.physics.pixels_per_meter)
    y_phys = maybe_scale_to_meters(y_smooth, cfg.physics.pixels_per_meter)

    derivs = compute_derivatives(x_phys, y_phys, t)

    # Physics fitting
    fit_results = {}
    if cfg.physics.fit.lower() == "projectile":
        # only fit on rows without NaNs
        mask = np.isfinite(x_phys) & np.isfinite(y_phys)
        if mask.sum() >= 5:
            fit = fit_projectile(t[mask], x_phys[mask], y_phys[mask], cfg.physics.y_axis_down)
            fit_results["projectile"] = fit
        else:
            fit_results["projectile"] = None
    # TODO: implement other models (linear, freefall, harmonic) if needed

    # Build output table
    out = pd.DataFrame({
        "frame": df["frame"],
        "time_s": t,
        "x_px": x_px,
        "y_px": y_px,
        "x_smooth_px": x_smooth,
        "y_smooth_px": y_smooth if cfg.physics.pixels_per_meter is None else df["y_px"],  # smoothed in px-space
        "x_phys": x_phys,
        "y_phys": y_phys,
        "vx_phys": derivs["vx"],
        "vy_phys": derivs["vy"],
        "ax_phys": derivs["ax"],
        "ay_phys": derivs["ay"],
        "speed_phys": derivs["speed"],
        "accel_phys": derivs["accel"],
    })

    # Add fit predictions if any
    if "projectile" in fit_results and fit_results["projectile"] is not None:
        fit = fit_results["projectile"]
        out["x_fit_phys"] = np.interp(t, t[np.isfinite(x_phys)], fit["x_fit"])
        out["y_fit_phys"] = np.interp(t, t[np.isfinite(y_phys)], fit["y_fit"])
        # store parameters in a sidecar JSON too
        params = {k: float(v) if np.isscalar(v) else None for k, v in fit.items()
                  if k in ["a", "b", "c", "d", "e", "g_px"]}
        with open(outdir / "fit_params_projectile.json", "w") as f:
            json.dump(params, f, indent=2)

    # Save CSV
    csv_path = outdir / cfg.output.csv_name
    out.to_csv(csv_path, index=False)

    # Plots
    if cfg.output.plots:
        # x(t), y(t)
        plt.figure()
        plt.plot(t, out["x_phys"] if cfg.physics.pixels_per_meter else out["x_px"], label="x")
        plt.xlabel("t [s]")
        plt.ylabel("x [m]" if cfg.physics.pixels_per_meter else "x [px]")
        plt.legend()
        plt.tight_layout()
        plt.savefig(outdir / f"{cfg.output.plots_prefix}_x_t.png", dpi=200)
        plt.close()

        plt.figure()
        plt.plot(t, out["y_phys"] if cfg.physics.pixels_per_meter else out["y_px"], label="y")
        if "y_fit_phys" in out.columns:
            plt.plot(t, out["y_fit_phys"], linestyle="--", label="y_fit")
        plt.xlabel("t [s]")
        plt.ylabel("y [m]" if cfg.physics.pixels_per_meter else "y [px]")
        plt.legend()
        plt.tight_layout()
        plt.savefig(outdir / f"{cfg.output.plots_prefix}_y_t.png", dpi=200)
        plt.close()

        # speed(t)
        plt.figure()
        plt.plot(t, out["speed_phys"], label="speed")
        plt.xlabel("t [s]")
        plt.ylabel("speed [m/s]" if cfg.physics.pixels_per_meter else "speed [px/s]")
        plt.legend()
        plt.tight_layout()
        plt.savefig(outdir / f"{cfg.output.plots_prefix}_speed_t.png", dpi=200)
        plt.close()

    print(f"[OK] Saved CSV: {csv_path}")
    if cfg.output.annotated_video:
        print(f"[OK] Saved annotated video: {outdir / cfg.output.annotated_name}")
    if cfg.output.plots:
        print(f"[OK] Saved plots to: {outdir}")


if __name__ == "__main__":
    main()
