from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any


TEMPLATE_HEADER = """#!/usr/bin/env python3
import argparse, json
from pathlib import Path
import cv2, numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DETECTION_METHOD = "__DETECTOR__"
HSV_LOWER = __HSV_LOWER__
HSV_UPPER = __HSV_UPPER__
MOTION_CLASS = "__MOTION__"
MIN_AREA = __MIN_AREA__
MORPH_KERNEL = __MORPH_KERNEL__

def detect_hsv(frame_bgr):
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array(HSV_LOWER, dtype=np.uint8), np.array(HSV_UPPER, dtype=np.uint8))
    if MORPH_KERNEL > 1:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (MORPH_KERNEL, MORPH_KERNEL))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    c = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(c) < MIN_AREA:
        return None
    M = cv2.moments(c)
    if M["m00"] <= 0: return None
    return int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"])

def detect_threshold(frame_bgr):
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    c = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(c) < MIN_AREA:
        return None
    M = cv2.moments(c)
    if M["m00"] <= 0: return None
    return int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"])

def detect_hough(frame_bgr):
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=20,
                               param1=100, param2=30, minRadius=3, maxRadius=0)
    if circles is None: return None
    circles = np.uint16(np.around(circles))
    x, y, r = circles[0][0]
    return int(x), int(y)

def dispatch(frame_bgr):
    if DETECTION_METHOD == "hsv":
        return detect_hsv(frame_bgr)
    elif DETECTION_METHOD == "hough":
        return detect_hough(frame_bgr)
    else:
        return detect_threshold(frame_bgr)

def smooth_series(a, window=9, poly=None):
    # Simple moving average fallback; avoids SciPy requirement inside generated code.
    w = max(3, window if window % 2 == 1 else window + 1)
    if len(a) < w:
        return a.copy()
    k = np.ones(w)/w
    return np.convolve(a, k, mode='same')

def fit_linear(t, y):
    A = np.vstack([t, np.ones_like(t)]).T
    m, b = np.linalg.lstsq(A, y, rcond=None)[0]
    return m, b, m*t + b

def fit_quadratic(t, y):
    A = np.vstack([t**2, t, np.ones_like(t)]).T
    a, b, c = np.linalg.lstsq(A, y, rcond=None)[0]
    return a, b, c, a*t*t + b*t + c

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()
    outdir = Path(args.outdir)
    (outdir / "plots").mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open video: {args.video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(outdir / 'annotated.mp4'), fourcc, fps, (W, H))

    records = []
    trail = []
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret: break
        t = idx / fps
        pos = dispatch(frame)
        if pos is not None:
            cx, cy = pos
            records.append({"frame": idx, "time_s": t, "x_px": cx, "y_px": cy})
            trail.append((int(cx), int(cy)))
            if len(trail) > 50: trail = trail[-50:]
            cv2.circle(frame, (int(cx), int(cy)), 5, (0,255,0), -1)
        else:
            records.append({"frame": idx, "time_s": t, "x_px": np.nan, "y_px": np.nan})
        for i in range(1, len(trail)):
            cv2.line(frame, trail[i-1], trail[i], (255,0,0), 2)
        cv2.putText(frame, f"frame={idx} t={t:.3f}s", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (50,220,50), 2)
        writer.write(frame)
        idx += 1
    cap.release(); writer.release()

    df = pd.DataFrame.from_records(records)
    df.sort_values('frame', inplace=True); df.reset_index(drop=True, inplace=True)

    # Smooth simple moving average
    x = df['x_px'].to_numpy(dtype=float)
    y = df['y_px'].to_numpy(dtype=float)
    t = df['time_s'].to_numpy(dtype=float)
    # Interpolate NaNs linearly for short gaps
    for col in ['x_px','y_px']:
        s = df[col].astype(float)
        if s.isna().all():
            continue
        df[col] = s.interpolate(limit_direction='both')
    x = df['x_px'].to_numpy(dtype=float)
    y = df['y_px'].to_numpy(dtype=float)
    x_s = smooth_series(x, window=9)
    y_s = smooth_series(y, window=9)

    # Fit based on motion class
    fit_params = {}
    if MOTION_CLASS in ("projectile", "free_fall"):
        d, e, x_fit = fit_linear(t, x_s)
        a, b, c, y_fit = fit_quadratic(t, y_s)
        fit_params.update({"x: d,e": [float(d), float(e)], "y: a,b,c": [float(a), float(b), float(c)]})
        df['x_fit_px'] = x_fit
        df['y_fit_px'] = y_fit
    elif MOTION_CLASS == "constant_velocity":
        m1, b1, x_fit = fit_linear(t, x_s)
        m2, b2, y_fit = fit_linear(t, y_s)
        fit_params.update({"x: m,b": [float(m1), float(b1)], "y: m,b": [float(m2), float(b2)]})
        df['x_fit_px'] = x_fit
        df['y_fit_px'] = y_fit
    elif MOTION_CLASS == "uniform_acceleration_1d":
        # PCA axis then 1D quadratic
        pts = np.column_stack([x_s, y_s])
        mu = np.nanmean(pts, axis=0)
        pts_c = pts - mu
        cov = np.cov(pts_c.T)
        w, v = np.linalg.eig(cov)
        axis = v[:, np.argmax(w)]  # principal axis
        s = pts_c @ axis
        a1, b1, c1, s_fit = fit_quadratic(t, s)
        # back-project fit to x,y for convenience
        xy_fit = np.outer(s_fit, axis) + mu
        df['x_fit_px'] = xy_fit[:,0]
        df['y_fit_px'] = xy_fit[:,1]
        fit_params.update({"axis": axis.tolist(), "s: a,b,c": [float(a1), float(b1), float(c1)]})
    else:
        # default: try linear for both
        m1, b1, x_fit = fit_linear(t, x_s)
        m2, b2, y_fit = fit_linear(t, y_s)
        df['x_fit_px'] = x_fit
        df['y_fit_px'] = y_fit
        fit_params.update({"x: m,b": [float(m1), float(b1)], "y: m,b": [float(m2), float(b2)]})

    # Save CSV
    df.to_csv(outdir / 'measurements.csv', index=False)

    # Plots
    plt.figure(); plt.plot(t, x, label='x'); plt.plot(t, df['x_fit_px'], '--', label='x_fit'); plt.legend(); plt.tight_layout(); plt.savefig((outdir / 'plots' / 'x_t.png'), dpi=150); plt.close()
    plt.figure(); plt.plot(t, y, label='y'); plt.plot(t, df['y_fit_px'], '--', label='y_fit'); plt.legend(); plt.tight_layout(); plt.savefig((outdir / 'plots' / 'y_t.png'), dpi=150); plt.close()

    # Fit params json
    (outdir / 'fit_params.json').write_text(json.dumps(fit_params, indent=2))

if __name__ == '__main__':
    main()
"""


@dataclass
class SimpleCodeGenerator:
    def generate(self, video_path: Path, prompt: str, gen_dir: Path, options: Dict[str, Any] | None = None) -> Path:
        options = options or {}
        motion = self._infer_motion_class(prompt)
        detector, hsv_lower, hsv_upper = self._infer_detector_and_color(prompt, options)
        content = TEMPLATE_HEADER
        content = content.replace("__DETECTOR__", detector)
        content = content.replace("__HSV_LOWER__", repr(hsv_lower))
        content = content.replace("__HSV_UPPER__", repr(hsv_upper))
        content = content.replace("__MOTION__", motion)
        content = content.replace("__MIN_AREA__", repr(options.get("min_area", 50)))
        content = content.replace("__MORPH_KERNEL__", repr(options.get("morph_kernel", 5)))
        out_path = gen_dir / "analyzer.py"
        out_path.write_text(content)
        return out_path

    def _infer_motion_class(self, prompt: str) -> str:
        p = prompt.lower()
        # Allow explicit tag override: [motion_class=...]
        if "[motion_class=" in p:
            try:
                s = p.split("[motion_class=",1)[1]
                tag = s.split("]",1)[0].strip()
                if tag in {"projectile","free_fall","constant_velocity","uniform_acceleration_1d"}:
                    return tag
            except Exception:
                pass
        if any(k in p for k in ["projectile", "parabolic", "throw", "ballistic"]):
            return "projectile"
        if any(k in p for k in ["free fall", "falling", "drop", "freefall"]):
            return "free_fall"
        if any(k in p for k in ["uniform acceleration", "accelerated", "acceleration 1d"]):
            return "uniform_acceleration_1d"
        if any(k in p for k in ["constant velocity", "linear motion", "uniform linear"]):
            return "constant_velocity"
        return "constant_velocity"

    def _infer_detector_and_color(self, prompt: str, options: Dict[str, Any]):
        # Detector selection
        detector = options.get("detector") or "hsv"
        # HSV bounds from options or prompt color keywords
        if options.get("hsv_lower") and options.get("hsv_upper"):
            return detector, options["hsv_lower"], options["hsv_upper"]
        p = prompt.lower()
        # Simple presets
        presets = {
            "blue": ([100, 80, 50], [130, 255, 255]),
            "red": ([0, 120, 120], [10, 255, 255]),
            "green": ([40, 40, 40], [85, 255, 255]),
            "yellow": ([20, 80, 80], [35, 255, 255]),
        }
        for color, hsv in presets.items():
            if color in p:
                return detector, hsv[0], hsv[1]
        # default to blue
        return detector, presets["blue"][0], presets["blue"][1]
