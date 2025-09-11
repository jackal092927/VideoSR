from pathlib import Path
import cv2
import pandas as pd

from .tracking_lite import track_centroid_hsv
from .smooth import smooth_and_derivatives


def run_extraction(video_path: str, cfg: dict):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    if cfg.get("sampling", {}).get("fps_override"):
        fps = float(cfg["sampling"]["fps_override"])
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    mode = cfg.get("mode", "lite").lower()
    if mode not in ["lite", "open_vocab"]:
        raise ValueError("mode must be 'lite' or 'open_vocab'")

    # --- 1) TRACK ---
    tracks = track_centroid_hsv(cap, cfg)  # list of dicts per frame
    cap.release()

    # --- 2) SMOOTH & DERIVE ---
    df = pd.DataFrame(tracks)  # columns: [frame, t, cx, cy, area, ok]
    df = smooth_and_derivatives(df, fps, cfg.get("savgol", {}))

    meta = {"fps": fps, "width": width, "height": height, "mode": mode}
    return df[["t", "x", "y", "vx", "vy", "ax", "ay", "area", "ok"]], meta