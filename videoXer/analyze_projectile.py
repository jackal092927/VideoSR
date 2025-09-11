#!/usr/bin/env python3
"""
Projectile Motion Analyzer (noise-robust, per-frame measurements)

This script loads a video of a (red) projectile, tracks the object frame-by-frame,
and produces a measurements CSV with time, position (raw & smoothed), velocity,
acceleration, and basic model fits. It also generates diagnostic plots and an
annotated video overlaying detections and the fitted parabola.

Dependencies (Python 3.9+):
  pip install opencv-python numpy pandas scipy matplotlib

Usage:
  python analyze_projectile.py --video "/mnt/data/Projectile_noisyTrajectory_1756271692.mp4"
  
  # Or with custom output paths:
  python analyze_projectile.py \
    --video "/mnt/data/Projectile_noisyTrajectory_1756271692.mp4" \
    --out_csv "/mnt/data/projectile_measurements.csv" \
    --out_video "/mnt/data/projectile_annotated.mp4" \
    --out_plot "/mnt/data/projectile_plots.png"

Notes:
- Color-based detection is tuned for a red ball. Adjust HSV ranges if needed.
- If frames are missed, a Kalman filter bridges short gaps.
- Smoothing uses Savitzky–Golay; window auto-tuned to FPS.
"""

import argparse
from dataclasses import dataclass, asdict
from typing import Optional, Tuple, List
import os
from datetime import datetime

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from math import isnan

try:
    from scipy.signal import savgol_filter
    from scipy.optimize import curve_fit
except Exception as e:
    raise RuntimeError("This script requires scipy. Please `pip install scipy`.") from e


# --------------------------
# Configuration Dataclass
# --------------------------

@dataclass
class Config:
    video_path: str
    out_csv: str
    out_video: Optional[str] = None
    out_plot: Optional[str] = None

    # Detection parameters
    # HSV ranges for "red" (two ranges because red wraps hue)
    lower_red1: Tuple[int, int, int] = (0, 80, 60)
    upper_red1: Tuple[int, int, int] = (10, 255, 255)
    lower_red2: Tuple[int, int, int] = (170, 80, 60)
    upper_red2: Tuple[int, int, int] = (179, 255, 255)

    morph_kernel: int = 5         # morphology kernel size
    min_area_px: int = 20         # min contour area to accept detection

    # Kalman filter (constant-velocity model) to bridge short gaps
    use_kalman: bool = True
    kalman_process_noise: float = 1e-2
    kalman_meas_noise: float = 1e-1
    kalman_error_post: float = 1.0

    # Smoothing & derivatives
    use_savgol: bool = True
    savgol_window_seconds: float = 0.15   # window ~15% second; will be converted to odd frames
    savgol_polyorder: int = 2

    # Drawing
    trail_max: int = 200  # number of trail points to draw

    # RANSAC for robust quadratic fit y(t) = a t^2 + b t + c
    use_ransac: bool = True
    ransac_trials: int = 200
    ransac_sample: int = 20   # number of points sampled each trial
    ransac_inlier_thresh: float = 5.0  # px error threshold in y

    # Optional ROI mask path (white=ROI, black=ignore). If provided, restricts search.
    roi_mask_path: Optional[str] = None


# --------------------------
# Utilities
# --------------------------

def auto_savgol_window(n_frames: int, fps: float, seconds: float) -> int:
    if not seconds or seconds <= 0 or fps <= 0:
        return 5 if n_frames >= 5 else max(3, n_frames | 1)
    win = int(max(5, round(seconds * fps)))
    if win % 2 == 0:
        win += 1
    win = min(win, n_frames if n_frames % 2 == 1 else n_frames - 1)
    return max(5, win)


def parabola_t(t, a, b, c):
    """y(t) = a t^2 + b t + c"""
    return a * t * t + b * t + c


def parabola_x(x, A, B, C):
    """y(x) = A x^2 + B x + C"""
    return A * x * x + B * x + C


def fit_parabola_time(t, y):
    """Least-squares fit y(t) = a t^2 + b t + c"""
    popt, pcov = curve_fit(parabola_t, t, y)
    a, b, c = popt
    residuals = y - parabola_t(t, *popt)
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return popt, r2


def fit_parabola_time_ransac(t, y, trials=200, sample=20, inlier_thresh=5.0):
    rng = np.random.default_rng(42)
    n = len(t)
    best_inliers = None
    best_popt = None

    if n < max(3, sample):
        return fit_parabola_time(t, y)

    idx_all = np.arange(n)
    for _ in range(trials):
        idx = rng.choice(idx_all, size=min(sample, n), replace=False)
        try:
            popt, _ = curve_fit(parabola_t, t[idx], y[idx])
        except Exception:
            continue
        y_pred = parabola_t(t, *popt)
        err = np.abs(y - y_pred)
        inliers = err < inlier_thresh
        if best_inliers is None or inliers.sum() > best_inliers.sum():
            best_inliers = inliers
            best_popt = popt

    if best_inliers is None or best_inliers.sum() < 3:
        return fit_parabola_time(t, y)

    popt, _ = curve_fit(parabola_t, t[best_inliers], y[best_inliers])
    residuals = y[best_inliers] - parabola_t(t[best_inliers], *popt)
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((y[best_inliers] - np.mean(y[best_inliers])) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return popt, r2


def fit_parabola_space(x, y):
    """Least-squares fit y(x) = A x^2 + B x + C"""
    popt, _ = curve_fit(parabola_x, x, y)
    A, B, C = popt
    residuals = y - parabola_x(x, *popt)
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return popt, r2


def load_mask(mask_path: Optional[str], shape: Tuple[int, int]) -> Optional[np.ndarray]:
    if not mask_path:
        return None
    m = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if m is None:
        return None
    if m.shape != shape[:2]:
        m = cv2.resize(m, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    return (m > 127).astype(np.uint8)


# --------------------------
# Kalman Filter Helper
# --------------------------

class SimpleKalman:
    """
    Constant-velocity Kalman for (x, y).
    State: [x, y, vx, vy]
    Measurement: [x, y]
    """
    def __init__(self, dt: float, process_noise: float, meas_noise: float, error_post: float):
        self.kf = cv2.KalmanFilter(4, 2, 0)
        self.kf.transitionMatrix = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ], dtype=np.float32)
        self.kf.measurementMatrix = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
        ], dtype=np.float32)
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * process_noise
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * meas_noise
        self.kf.errorCovPost = np.eye(4, dtype=np.float32) * error_post
        self.initialized = False

    def init_state(self, x, y):
        self.kf.statePost = np.array([[x], [y], [0.0], [0.0]], dtype=np.float32)
        self.initialized = True

    def predict(self):
        return self.kf.predict()

    def correct(self, x, y):
        meas = np.array([[np.float32(x)], [np.float32(y)]], dtype=np.float32)
        return self.kf.correct(meas)


# --------------------------
# Detection
# --------------------------

def detect_red_blob(frame_bgr: np.ndarray, cfg: Config, roi_mask: Optional[np.ndarray]) -> Optional[Tuple[int, int, float]]:
    """Returns (cx, cy, area) or None"""
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)

    mask1 = cv2.inRange(hsv, np.array(cfg.lower_red1), np.array(cfg.upper_red1))
    mask2 = cv2.inRange(hsv, np.array(cfg.lower_red2), np.array(cfg.upper_red2))
    mask = cv2.bitwise_or(mask1, mask2)

    if roi_mask is not None:
        mask = cv2.bitwise_and(mask, mask, mask=roi_mask)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (cfg.morph_kernel, cfg.morph_kernel))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None

    # choose the most circular-ish large blob
    best = None
    best_score = -np.inf
    for c in cnts:
        area = cv2.contourArea(c)
        if area < cfg.min_area_px:
            continue
        (x, y), radius = cv2.minEnclosingCircle(c)
        if radius <= 0:
            continue
        circ = (4 * np.pi * area) / (cv2.arcLength(c, True) ** 2 + 1e-6)  # shape circularity score
        score = area + 50.0 * circ
        if score > best_score:
            M = cv2.moments(c)
            if M["m00"] > 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                best = (cx, cy, float(area))
                best_score = score

    return best


# --------------------------
# Main Processing
# --------------------------

def analyze(cfg: Config):
    cap = cv2.VideoCapture(cfg.video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {cfg.video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 1e-6 or np.isnan(fps):
        # fallback to a common rate if video is missing metadata
        fps = 30.0

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    roi_mask = load_mask(cfg.roi_mask_path, (height, width))

    # Prepare writer
    writer = None
    if cfg.out_video:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(cfg.out_video, fourcc, fps, (width, height))

    # Kalman
    kalman = SimpleKalman(
        dt=1.0 / fps,
        process_noise=cfg.kalman_process_noise,
        meas_noise=cfg.kalman_meas_noise,
        error_post=cfg.kalman_error_post,
    ) if cfg.use_kalman else None

    rows = []
    trail: List[Tuple[int, int]] = []

    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t = frame_idx / fps

        det = detect_red_blob(frame, cfg, roi_mask)

        if det is not None:
            cx, cy, area = det
            if kalman is not None and not kalman.initialized:
                kalman.init_state(cx, cy)
            if kalman is not None:
                kalman.predict()
                kalman.correct(cx, cy)
                state = kalman.kf.statePost.flatten()
                kx, ky, kvx, kvy = map(float, state)
            else:
                kx, ky, kvx, kvy = float(cx), float(cy), np.nan, np.nan

            detected = True
        else:
            if kalman is not None and kalman.initialized:
                pred = kalman.predict()
                kx, ky, kvx, kvy = map(float, pred.flatten())
            else:
                kx = ky = kvx = kvy = np.nan
            detected = False

        rows.append({
            "frame": frame_idx,
            "time_s": t,
            "x_px": float(cx) if det is not None else np.nan,
            "y_px": float(cy) if det is not None else np.nan,
            "x_est": kx,
            "y_est": ky,
            "vx_est": kvx,
            "vy_est": kvy,
            "detected": bool(detected),
        })

        # Draw
        if writer is not None:
            disp = frame.copy()
            if not isnan(kx) and not isnan(ky):
                cv2.circle(disp, (int(round(kx)), int(round(ky))), 6, (0, 0, 255), -1)
                trail.append((int(round(kx)), int(round(ky))))
                trail = trail[-cfg.trail_max:]

            # draw trail
            for i in range(1, len(trail)):
                cv2.line(disp, trail[i-1], trail[i], (0, 255, 0), 2)

            # basic HUD text
            cv2.putText(disp, f"t={t:.3f}s  fps={fps:.2f}", (10, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

            writer.write(disp)

        frame_idx += 1

    cap.release()
    if writer is not None:
        writer.release()

    df = pd.DataFrame(rows)

    # Prefer estimated kalman positions (bridges gaps); fall back to raw when avail
    x = df["x_est"].to_numpy().copy()
    y = df["y_est"].to_numpy().copy()
    raw_x = df["x_px"].to_numpy()
    raw_y = df["y_px"].to_numpy()
    t = df["time_s"].to_numpy()

    # Where Kalman is NaN but raw exists, fill
    nan_x = np.isnan(x)
    nan_y = np.isnan(y)
    x[nan_x & ~np.isnan(raw_x)] = raw_x[nan_x & ~np.isnan(raw_x)]
    y[nan_y & ~np.isnan(raw_y)] = raw_y[nan_y & ~np.isnan(raw_y)]

    # Interpolate small gaps linearly for smoothing stability
    def interp_nans(arr):
        arr2 = arr.copy()
        n = len(arr2)
        idx = np.arange(n)
        mask = np.isfinite(arr2)
        if mask.sum() >= 2:
            arr2[~mask] = np.interp(idx[~mask], idx[mask], arr2[mask])
        return arr2

    xi = interp_nans(x)
    yi = interp_nans(y)

    # Savitzky–Golay smoothing
    if cfg.use_savgol:
        win = auto_savgol_window(len(df), fps, cfg.savgol_window_seconds)
        try:
            xs = savgol_filter(xi, window_length=win, polyorder=cfg.savgol_polyorder, mode="interp")
            ys = savgol_filter(yi, window_length=win, polyorder=cfg.savgol_polyorder, mode="interp")
        except ValueError:
            # If too few points, fall back to interpolated
            xs, ys = xi, yi
    else:
        xs, ys = xi, yi

    # Derivatives (central differences)
    dt = np.gradient(t)
    vxs = np.gradient(xs, t, edge_order=2)
    vys = np.gradient(ys, t, edge_order=2)
    axs = np.gradient(vxs, t, edge_order=2)
    ays = np.gradient(vys, t, edge_order=2)

    # Fits
    finite_mask = np.isfinite(ys) & np.isfinite(t)
    tt = t[finite_mask]
    yy = ys[finite_mask]

    if len(tt) >= 5:
        if cfg.use_ransac:
            (a_t, b_t, c_t), r2_t = fit_parabola_time_ransac(tt, yy,
                                                              trials=cfg.ransac_trials,
                                                              sample=cfg.ransac_sample,
                                                              inlier_thresh=cfg.ransac_inlier_thresh)
        else:
            (a_t, b_t, c_t), r2_t = fit_parabola_time(tt, yy)
    else:
        a_t = b_t = c_t = r2_t = np.nan

    finite_mask2 = np.isfinite(xs) & np.isfinite(ys)
    xx2 = xs[finite_mask2]
    yy2 = ys[finite_mask2]
    if len(xx2) >= 5:
        (A_x, B_x, C_x), r2_x = fit_parabola_space(xx2, yy2)
    else:
        A_x = B_x = C_x = r2_x = np.nan

    # Estimated gravity in px/s^2 from y(t) = a t^2 + b t + c => y'' = 2a
    g_px = 2.0 * a_t if np.isfinite(a_t) else np.nan

    # Assemble output DataFrame
    out = df.copy()
    out["x_smooth_px"] = xs
    out["y_smooth_px"] = ys
    out["vx_px_s"] = vxs
    out["vy_px_s"] = vys
    out["ax_px_s2"] = axs
    out["ay_px_s2"] = ays
    out["fit_time_a"] = a_t
    out["fit_time_b"] = b_t
    out["fit_time_c"] = c_t
    out["fit_time_r2"] = r2_t
    out["fit_space_A"] = A_x
    out["fit_space_B"] = B_x
    out["fit_space_C"] = C_x
    out["fit_space_r2"] = r2_x
    out["g_est_px_s2"] = g_px
    out["fps"] = fps

    # Save CSV
    Path(cfg.out_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(cfg.out_csv, index=False)

    # Plots
    if cfg.out_plot:
        fig1 = plt.figure(figsize=(10, 6))
        # Trajectory
        plt.scatter(xs, ys, s=8, label="trajectory (smoothed)")
        if np.isfinite(A_x):
            xs_plot = np.linspace(np.nanmin(xs), np.nanmax(xs), 200)
            ys_plot = parabola_x(xs_plot, A_x, B_x, C_x)
            plt.plot(xs_plot, ys_plot, linewidth=2, label=f"fit y(x) R²={r2_x:.3f}")
        plt.gca().invert_yaxis()
        plt.title("Projectile trajectory (pixels)\nOrigin at top-left; y increases downward")
        plt.xlabel("x [px]")
        plt.ylabel("y [px]")
        plt.legend()
        fig1.tight_layout()

        fig2 = plt.figure(figsize=(10, 6))
        plt.plot(t, ys, linewidth=1.5, label="y_smooth [px]")
        if np.isfinite(a_t):
            ys_tfit = parabola_t(t, a_t, b_t, c_t)
            plt.plot(t, ys_tfit, linewidth=2, label=f"fit y(t) a={a_t:.3g}, b={b_t:.3g}, c={c_t:.3g}, R²={r2_t:.3f}")
        plt.gca().invert_yaxis()
        plt.xlabel("time [s]"); plt.ylabel("y [px]")
        plt.title(f"y(t) and quadratic fit — g_est ≈ {g_px:.2f} px/s²")
        plt.legend()
        fig2.tight_layout()

        fig3 = plt.figure(figsize=(10, 6))
        plt.plot(t, vxs, label="vx [px/s]")
        plt.plot(t, vys, label="vy [px/s]")
        plt.xlabel("time [s]"); plt.ylabel("velocity [px/s]")
        plt.title("Velocity vs time")
        plt.legend()
        fig3.tight_layout()

        fig4 = plt.figure(figsize=(10, 6))
        plt.plot(t, axs, label="ax [px/s²]")
        plt.plot(t, ays, label="ay [px/s²]")
        plt.xlabel("time [s]"); plt.ylabel("acceleration [px/s²]")
        plt.title("Acceleration vs time")
        plt.legend()
        fig4.tight_layout()

        fig1.savefig(cfg.out_plot, dpi=150)
        # Save additional plots alongside
        base = Path(cfg.out_plot)
        fig2.savefig(str(base.with_name(base.stem + "_y_of_t.png")), dpi=150)
        fig3.savefig(str(base.with_name(base.stem + "_vel.png")), dpi=150)
        fig4.savefig(str(base.with_name(base.stem + "_acc.png")), dpi=150)

        plt.close(fig1); plt.close(fig2); plt.close(fig3); plt.close(fig4)

    # If annotated video requested, draw final fitted parabola overlay by re-writing quick preview of last frame
    # (Full overlay per-frame would require another pass; current per-frame overlay already draws track).
    return {
        "n_frames": n_frames,
        "fps": fps,
        "csv": cfg.out_csv,
        "video": cfg.out_video,
        "plot": cfg.out_plot,
        "g_px_s2": g_px,
        "r2_time": r2_t,
        "r2_space": r2_x,
    }


def generate_default_paths(video_path: str):
    """Generate default output paths based on input video name and timestamp."""
    # Create output directory if it doesn't exist
    output_dir = "./output"
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract video name without extension
    video_name = Path(video_path).stem
    
    # Generate timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Generate base filename
    base_filename = f"{video_name}_{timestamp}"
    
    return {
        "csv": os.path.join(output_dir, f"{base_filename}.csv"),
        "video": os.path.join(output_dir, f"{base_filename}_annotated.mp4"),
        "plot": os.path.join(output_dir, f"{base_filename}_trajectory.png")
    }


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--video", required=True, help="Path to input video")
    p.add_argument("--out_csv", default=None, help="Path to output CSV (default: auto-generated under ./output)")
    p.add_argument("--out_video", default=None, help="Path to annotated MP4 to write (default: auto-generated under ./output)")
    p.add_argument("--out_plot", default=None, help="Path to main plot PNG to write (default: auto-generated under ./output)")
    p.add_argument("--roi_mask", default=None, help="Optional binary mask image to restrict search area")
    return p.parse_args()


def main():
    args = parse_args()
    
    # Generate default paths if not provided
    default_paths = generate_default_paths(args.video)
    
    cfg = Config(
        video_path=args.video,
        out_csv=args.out_csv or default_paths["csv"],
        out_video=args.out_video or default_paths["video"],
        out_plot=args.out_plot or default_paths["plot"],
        roi_mask_path=args.roi_mask,
    )
    stats = analyze(cfg)
    print("=== Analysis Summary ===")
    for k, v in stats.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
