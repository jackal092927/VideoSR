from __future__ import annotations

from typing import Dict, Any
import cv2
import numpy as np
import pandas as pd


def render_overlay(video: str, df: pd.DataFrame, out_mp4, cfg: Dict[str, Any]):
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(out_mp4), fourcc, fps, (W, H))

    trail = []
    show_fit = bool(cfg.get('show_fit', True))
    trail_len = int(cfg.get('trail_len', 60))
    dot_radius = int(cfg.get('dot_radius', 8))

    # Precompute full track and fit polylines for clarity
    pts_track = []
    if {'x_px','y_px'}.issubset(df.columns):
        for _, row in df.iterrows():
            x, y = row['x_px'], row['y_px']
            if np.isfinite(x) and np.isfinite(y):
                pts_track.append((int(max(0, min(W-1, x))), int(max(0, min(H-1, y)))))

    pts_fit = []
    if {'x_fit_px','y_fit_px'}.issubset(df.columns):
        for _, row in df.iterrows():
            x, y = row['x_fit_px'], row['y_fit_px']
            if np.isfinite(x) and np.isfinite(y):
                pts_fit.append((int(max(0, min(W-1, x))), int(max(0, min(H-1, y)))))

    for i in range(len(df)):
        ok, frame = cap.read()
        if not ok:
            break
        row = df.iloc[i]

        # Draw track point with high-contrast styling
        if np.isfinite(row.get('x_px', np.nan)) and np.isfinite(row.get('y_px', np.nan)):
            x = int(max(0, min(W-1, row['x_px'])))
            y = int(max(0, min(H-1, row['y_px'])))
            pt = (x, y)
            trail.append(pt)
            if len(trail) > trail_len:
                trail = trail[-trail_len:]
            # Outer ring (yellow), inner dot (magenta), and crosshair
            cv2.circle(frame, pt, dot_radius+2, (0,255,255), 2)     # yellow ring
            cv2.circle(frame, pt, max(2, dot_radius//2), (255,0,255), -1)  # magenta center
            cv2.line(frame, (x-10, y), (x+10, y), (0,255,255), 1)
            cv2.line(frame, (x, y-10), (x, y+10), (0,255,255), 1)

        # Trail (bright yellow)
        for j in range(1, len(trail)):
            cv2.line(frame, trail[j-1], trail[j], (0,255,255), 2)

        # Optional fit point (cyan)
        if show_fit and 'x_fit_px' in df.columns and 'y_fit_px' in df.columns:
            fx, fy = row['x_fit_px'], row['y_fit_px']
            if np.isfinite(fx) and np.isfinite(fy):
                fx = int(max(0, min(W-1, fx)))
                fy = int(max(0, min(H-1, fy)))
                cv2.circle(frame, (fx, fy), 4, (255,255,0), 1)

        # Draw global polylines faintly for context
        if len(pts_track) > 1:
            for j in range(1, len(pts_track)):
                cv2.line(frame, pts_track[j-1], pts_track[j], (0, 200, 255), 1)
        if show_fit and len(pts_fit) > 1:
            for j in range(1, len(pts_fit)):
                cv2.line(frame, pts_fit[j-1], pts_fit[j], (255, 255, 0), 1)

        # HUD: frame and time, top-right for visibility
        t_val = row['time_s'] if 'time_s' in row else 0.0
        text = f"frame={i}  t={t_val:.3f}s"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.putText(frame, text, (W - tw - 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 3)
        cv2.putText(frame, text, (W - tw - 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

        writer.write(frame)

    cap.release(); writer.release()
