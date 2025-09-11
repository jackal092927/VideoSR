
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
analyze_gsam2_incline.py — Text-prompted tracking via Grounded-SAM-2, then physics on an inclined plane

Pipeline
--------
1) (Optional) Run Grounded-SAM-2 on a video with a text prompt (e.g., "toy car").
2) Load GSAM-2 results (masks or boxes) to extract per-frame centroids.
3) Detect (or accept manual) incline line; project centroids onto incline axis.
4) Smooth s(t), compute v(t), a(t); fit s(t)=s0+v0*t+0.5*a*t^2.
5) Save CSV + plots + an annotated video.

You can skip step (1) by passing --gsam2_results_dir that GSAM-2 already produced.

Example A (run GSAM-2 here)
---------------------------
python analyze_gsam2_incline.py \
  --video /path/to/video.mp4 \
  --prompt "toy car" \
  --gsam2_repo /path/to/Grounded-SAM-2 \
  --outdir ./gsam2_incline_out

Example B (use precomputed GSAM-2 results)
------------------------------------------
python analyze_gsam2_incline.py \
  --video /path/to/video.mp4 \
  --prompt "toy car" \
  --gsam2_results_dir /path/to/gsam2_results \
  --outdir ./gsam2_incline_out
"""

import argparse, sys, subprocess, json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List, Tuple, Dict

import cv2, numpy as np, pandas as pd

try:
    from scipy.signal import savgol_filter
except Exception:
    savgol_filter = None

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------- Config dataclasses ----------------

@dataclass
class GSAM2Config:
    run_gsam2: bool = True
    repo_dir: Optional[str] = None       # path to cloned Grounded-SAM-2 repo
    entry: Optional[str] = None          # optional explicit demo script path
    text_threshold: float = 0.25
    box_threshold: float = 0.25
    mask_output: bool = True             # request mask outputs if supported
    results_dir: Optional[str] = None    # if provided, skip running GSAM-2 and just read

@dataclass
class InclineConfig:
    auto_detect: bool = True
    manual_points: Optional[List[int]] = None  # [x1,y1,x2,y2] top->bottom
    # Edge/Hough for auto incline
    pre_blur_ksize: int = 5
    canny1: int = 60
    canny2: int = 140
    houghP_threshold: int = 70
    houghP_min_line_len: int = 80
    houghP_max_gap: int = 8
    min_angle_deg: float = 10.0
    max_angle_deg: float = 80.0

@dataclass
class TrackingConfig:
    smooth_method: str = "savgol"   # 'none' | 'savgol'
    smooth_window: int = 9
    smooth_poly: int = 2

@dataclass
class PhysicsConfig:
    pixels_per_meter: Optional[float] = None
    y_axis_down: bool = True

@dataclass
class OutputConfig:
    annotated_video: bool = True
    csv_name: str = "measurements.csv"
    annotated_name: str = "annotated.mp4"
    plots_prefix: str = "plot"

@dataclass
class AppConfig:
    gsam2: GSAM2Config = field(default_factory=GSAM2Config)
    incline: InclineConfig = field(default_factory=InclineConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    physics: PhysicsConfig = field(default_factory=PhysicsConfig)
    output: OutputConfig = field(default_factory=OutputConfig)


# ---------------- Utility ----------------

def load_config(path: Optional[Path]) -> AppConfig:
    if path is None:
        return AppConfig()
    with open(path, "r") as f:
        raw = json.load(f)

    def dc(cls, d):
        return cls(**{k: d.get(k, getattr(cls(), k)) for k in cls().__dict__.keys()})
    return AppConfig(
        gsam2=dc(GSAM2Config, raw.get("gsam2", {})),
        incline=dc(InclineConfig, raw.get("incline", {})),
        tracking=dc(TrackingConfig, raw.get("tracking", {})),
        physics=dc(PhysicsConfig, raw.get("physics", {})),
        output=dc(OutputConfig, raw.get("output", {})),
    )

def load_video(video_path: Path):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    N = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    return cap, float(fps), (W,H), N

def sample_background(video_path: Path, fps: float, N: int, stride: int = 5, samples: int = 60):
    cap = cv2.VideoCapture(str(video_path))
    frames = []
    idx = 0
    step = max(1, int(N / max(1, samples*stride)))
    while True:
        ret, frame = cap.read()
        if not ret: break
        if (idx % (stride*step)) == 0:
            frames.append(frame)
            if len(frames) >= samples: break
        idx += 1
    cap.release()
    if not frames: raise RuntimeError("No frames sampled for background")
    return np.median(np.stack(frames, axis=0), axis=0).astype(np.uint8)

def fold_angle_deg(a: float) -> float:
    a = abs(a)
    return 180-a if a>90 else a

def detect_incline_line(bg_bgr, inc: InclineConfig) -> Tuple[np.ndarray, np.ndarray, Tuple[int,int,int,int]]:
    gray = cv2.cvtColor(bg_bgr, cv2.COLOR_BGR2GRAY)
    if inc.pre_blur_ksize and inc.pre_blur_ksize%2==1 and inc.pre_blur_ksize>1:
        gray = cv2.GaussianBlur(gray, (inc.pre_blur_ksize, inc.pre_blur_ksize), 0)
    edges = cv2.Canny(gray, inc.canny1, inc.canny2)
    linesP = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=inc.houghP_threshold,
                             minLineLength=inc.houghP_min_line_len, maxLineGap=inc.houghP_max_gap)
    best=None
    if linesP is not None:
        for x1,y1,x2,y2 in linesP[:,0,:]:
            dx,dy = x2-x1, y2-y1
            angle = fold_angle_deg(np.degrees(np.arctan2(dy,dx)))
            if inc.min_angle_deg <= angle <= inc.max_angle_deg:
                length = float(np.hypot(dx,dy))
                if best is None or length > best[0]:
                    best=(length,(x1,y1,x2,y2))
    if best is None:
        raise RuntimeError("Could not detect incline; pass --manual_points x1 y1 x2 y2")
    x1,y1,x2,y2 = best[1]
    v = np.array([x2-x1, y2-y1], float)
    n = np.linalg.norm(v)
    if n < 1e-6:
        raise RuntimeError("Degenerate incline line")
    u = v / n
    # orient top->bottom (image y grows downward)
    x_top, y_top = (x1,y1) if y1 < y2 else (x2,y2)
    x0 = np.array([x_top, y_top], float)
    return u, x0, (x1,y1,x2,y2)

def band_mask(shape, u, x0, halfwidth):
    H,W = shape[:2]
    yy,xx = np.mgrid[0:H,0:W]
    P = np.stack([xx,yy],axis=-1).astype(float)
    n = np.array([-u[1], u[0]])
    dist = np.abs((P - x0) @ n)
    return dist <= halfwidth


# ---------------- Grounded-SAM-2 integration ----------------

def run_gsam2_if_needed(video_path: Path, prompt: str, cfg: GSAM2Config, outdir: Path) -> Path:
    """
    If cfg.results_dir is provided, just return that.
    Otherwise, attempt to locate a GSAM-2 demo script in cfg.repo_dir and run it.
    Return the results directory path that contains per-frame outputs.
    """
    if cfg.results_dir:
        results = Path(cfg.results_dir)
        if not results.exists():
            raise FileNotFoundError(f"Provided --gsam2_results_dir does not exist: {results}")
        return results

    if not cfg.run_gsam2:
        raise RuntimeError("GSAM-2 execution disabled and no --gsam2_results_dir given.")

    repo = Path(cfg.repo_dir) if cfg.repo_dir else None
    if (repo is None) or (not repo.exists()):
        raise FileNotFoundError("GSAM-2 repo_dir not found. Pass --gsam2_repo /path/to/Grounded-SAM-2")

    # Candidate demo scripts (common names in the repo; adjust if needed)
    candidates = []
    if cfg.entry:
        candidates.append(Path(cfg.entry))
    candidates += [
        repo / "demo" / "grounded_sam2_tracking_demo_custom_video_input_gd1.0_local_model.py",
        repo / "demo" / "grounded_sam2_tracking_demo_custom_video_input_gd15_cloudapi.py",
        repo / "demo" / "grounded_sam2_tracking_demo_custom_video_input.py",
    ]
    demo_script = next((p for p in candidates if p.exists()), None)
    if demo_script is None:
        raise FileNotFoundError(
            "Could not find a known GSAM-2 demo script in repo. "
            "Look for a demo like 'grounded_sam2_tracking_demo_custom_video_input_gd1.0_local_model.py' "
            "and pass its path with --gsam2_entry."
        )

    results_dir = outdir / "gsam2_results"
    results_dir.mkdir(parents=True, exist_ok=True)

    # Build a best-effort command. Adapt flags to your repo version if needed.
    cmd = [
        sys.executable, str(demo_script),
        "--input_video", str(video_path),
        "--text_prompt", prompt,
        "--output_dir", str(results_dir),
        "--box_threshold", str(cfg.box_threshold),
        "--text_threshold", str(cfg.text_threshold),
    ]
    if cfg.mask_output:
        cmd += ["--mask_output", "true"]

    print("[GSAM-2] Running:", " ".join(cmd))
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            "Grounded-SAM-2 demo execution failed. "
            "Open the repo README and ensure dependencies & script name/flags match your version."
        ) from e

    return results_dir

def load_gsam2_centroids(results_dir: Path, frame_count: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load per-frame mask(s) or boxes and compute centroids.
    Returns: frame_idx, cx, cy (NaN if missing)
    """
    # 1) Try masks/*.png
    mask_dirs = [
        results_dir / "masks",
        results_dir / "mask",
        results_dir / "segments",
        results_dir
    ]
    mask_paths = []
    for d in mask_dirs:
        if d.exists() and d.is_dir():
            cand = sorted([p for p in d.glob("*.png") if p.is_file()])
            if cand:
                mask_paths = cand
                break

    def centroid_from_mask(img_path: Path):
        m = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if m is None: return np.nan, np.nan
        _, mbin = cv2.threshold(m, 127, 255, cv2.THRESH_BINARY)
        cnts,_ = cv2.findContours(mbin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts: return np.nan, np.nan
        c = max(cnts, key=cv2.contourArea)
        if cv2.contourArea(c) < 5: return np.nan, np.nan
        M = cv2.moments(c)
        if M["m00"]<=0: return np.nan, np.nan
        cx = M["m10"]/M["m00"]; cy = M["m01"]/M["m00"]
        return float(cx), float(cy)

    if mask_paths:
        count = min(frame_count, len(mask_paths))
        idxs = np.arange(count)
        cxs = np.empty(count); cys = np.empty(count)
        cxs[:] = np.nan; cys[:] = np.nan
        for i in range(count):
            cx, cy = centroid_from_mask(mask_paths[i])
            cxs[i] = cx; cys[i] = cy
        return idxs, cxs, cys

    # 2) Try boxes JSON
    json_files = list(results_dir.glob("*.json"))
    box_json = None
    for p in json_files:
        if "track" in p.name or "result" in p.name or "detect" in p.name:
            box_json = p; break
    if box_json is None and json_files:
        box_json = json_files[0]

    if box_json is not None:
        data = json.loads(Path(box_json).read_text())
        cxs = []; cys = []
        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict):
                    if "cx" in item and "cy" in item:
                        cxs.append(float(item["cx"])); cys.append(float(item["cy"]))
                    elif "bbox" in item:
                        x1,y1,x2,y2 = item["bbox"]
                        cxs.append(0.5*(x1+x2)); cys.append(0.5*(y1+y2))
        elif isinstance(data, dict):
            keys = sorted([int(k) for k in data.keys() if str(k).isdigit()])
            for k in keys:
                v = data[str(k)]
                if isinstance(v, dict):
                    if "cx" in v and "cy" in v:
                        cxs.append(float(v["cx"])); cys.append(float(v["cy"]))
                    elif "bbox" in v:
                        x1,y1,x2,y2 = v["bbox"]
                        cxs.append(0.5*(x1+x2)); cys.append(0.5*(y1+y2))
        if cxs:
            idxs = np.arange(len(cxs))
            return idxs, np.array(cxs, float), np.array(cys, float)

    raise FileNotFoundError(
        f"Could not find masks or box JSON under: {results_dir}\n"
        "Please confirm your Grounded-SAM-2 output folder and pass it with --gsam2_results_dir."
    )


# ---------------- Core pipeline ----------------

def smooth_series(arr: np.ndarray, method: str, window: int, poly: int) -> np.ndarray:
    if method == "none":
        return arr.copy()
    if method == "savgol" and savgol_filter is not None and len(arr) >= max(window,5):
        w = window if window % 2 == 1 else window+1
        w = min(w, len(arr) - (1 - len(arr) % 2)) if len(arr) > 2 else 3
        w = max(5, w)
        p = min(poly, w-1)
        try:
            return savgol_filter(arr, window_length=w, polyorder=p, mode="interp")
        except Exception:
            return arr.copy()
    return arr.copy()

def project_point_on_line(pt: np.ndarray, u: np.ndarray, x0: np.ndarray) -> float:
    return float((pt - x0) @ u)

def main():
    ap = argparse.ArgumentParser(description="GSAM-2 prompted incline physics analyzer")
    ap.add_argument("--config", type=str, default=None)
    ap.add_argument("--video", type=str, required=True)
    ap.add_argument("--prompt", type=str, required=True, help='Text prompt, e.g., "toy car"')

    # GSAM-2 integration
    ap.add_argument("--gsam2_repo", type=str, default=None, help="Path to Grounded-SAM-2 repo")
    ap.add_argument("--gsam2_entry", type=str, default=None, help="Path to a specific GSAM-2 demo script")
    ap.add_argument("--gsam2_results_dir", type=str, default=None, help="Use existing results instead of running GSAM-2")
    ap.add_argument("--text_threshold", type=float, default=None)
    ap.add_argument("--box_threshold", type=float, default=None)

    # Incline
    ap.add_argument("--manual_points", type=float, nargs=4, default=None, help="x_top y_top x_bottom y_bottom")
    ap.add_argument("--auto_detect", type=str, default=None, help="override config.incline.auto_detect (true/false)")

    # Output
    ap.add_argument("--outdir", type=str, default="gsam2_incline_out")
    args = ap.parse_args()

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    cfg = load_config(Path(args.config) if args.config else None)

    # Override from CLI
    cfg.gsam2.repo_dir = args.gsam2_repo or cfg.gsam2.repo_dir
    cfg.gsam2.entry = args.gsam2_entry or cfg.gsam2.entry
    cfg.gsam2.results_dir = args.gsam2_results_dir or cfg.gsam2.results_dir
    if args.text_threshold is not None: cfg.gsam2.text_threshold = args.text_threshold
    if args.box_threshold is not None: cfg.gsam2.box_threshold = args.box_threshold
    if args.auto_detect is not None:
        cfg.incline.auto_detect = args.auto_detect.lower() in ("1","true","yes","y")
    if args.manual_points is not None:
        cfg.incline.manual_points = list(map(int, args.manual_points))

    # Load video, background, and timebase
    video_path = Path(args.video)
    cap, fps, (W,H), N = load_video(video_path)
    bg = sample_background(video_path, fps, N, stride=5, samples=60)
    times = np.arange(N) / fps

    # Run or load GSAM-2 results
    results_dir = run_gsam2_if_needed(video_path, args.prompt, cfg.gsam2, outdir)
    frame_idx, cx, cy = load_gsam2_centroids(Path(results_dir), N)

    # Align lengths to video frame count
    count = min(N, len(frame_idx))
    cx = cx[:count]; cy = cy[:count]; times = times[:count]

    # Incline (auto or manual)
    if not cfg.incline.auto_detect and cfg.incline.manual_points:
        x1,y1,x2,y2 = cfg.incline.manual_points
        v = np.array([x2-x1, y2-y1], float)
        u = v / (np.linalg.norm(v)+1e-9)
        x0 = np.array([x1,y1], float)
    else:
        u, x0, seg = detect_incline_line(bg, cfg.incline)

    # Project centroids onto incline; build dataframe
    s = np.full(count, np.nan, float)
    for i in range(count):
        if np.isfinite(cx[i]) and np.isfinite(cy[i]):
            s[i] = project_point_on_line(np.array([cx[i], cy[i]], float), u, x0)

    # Smooth + derivatives + fit
    s_smooth = smooth_series(s, cfg.tracking.smooth_method, cfg.tracking.smooth_window, cfg.tracking.smooth_poly)
    vs = np.gradient(pd.Series(s_smooth).fillna(method="ffill").fillna(method="bfill").to_numpy(), times, edge_order=2)
    acc = np.gradient(vs, times, edge_order=2)

    mask = np.isfinite(s_smooth) & np.isfinite(times)
    if mask.sum() >= 5:
        A = np.vstack([np.ones_like(times[mask]), times[mask], 0.5*(times[mask]**2)]).T
        sol,_res,_r,_s = np.linalg.lstsq(A, s_smooth[mask], rcond=None)
        s0,v0,a = sol
        s_fit = s0 + v0*times + 0.5*a*(times**2)
    else:
        s0=v0=a=np.nan
        s_fit = np.full_like(times, np.nan)

    # Optional scale to meters
    ppm = cfg.physics.pixels_per_meter
    df = pd.DataFrame({
        "frame": np.arange(count),
        "time_s": times,
        "cx_px": cx,
        "cy_px": cy,
        "s_px": s,
        "s_smooth_px": s_smooth,
        "v_along_px_s": vs,
        "a_along_px_s2": acc,
        "s_fit_px": s_fit
    })
    if ppm and ppm > 0:
        df["s_m"] = df["s_px"] / ppm
        df["v_along_m_s"] = df["v_along_px_s"] / ppm
        df["a_along_m_s2"] = df["a_along_px_s2"] / ppm
        df["s_fit_m"] = df["s_fit_px"] / ppm

    # Save CSV
    csv_path = outdir / cfg.output.csv_name
    df.to_csv(csv_path, index=False)

    # Annotated video (draw incline line and centroid)
    writer=None
    if cfg.output.annotated_video:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(outdir / cfg.output.annotated_name), fourcc, fps, (W,H))
        cap2 = cv2.VideoCapture(str(video_path))
        i=0
        while True:
            ret, frame = cap2.read()
            if not ret or i>=count: break
            # line
            p0 = tuple(np.int32(x0))
            p1 = tuple(np.int32(x0 + u*2000))
            cv2.line(frame, p0, p1, (0,255,0), 2)
            # centroid
            if np.isfinite(cx[i]) and np.isfinite(cy[i]):
                cv2.circle(frame, (int(cx[i]), int(cy[i])), 5, (0,0,255), -1)
                cv2.putText(frame, f"s={s[i]:.1f}px", (int(cx[i])+8,int(cy[i])-8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
            cv2.putText(frame, f"t={times[i]:.2f}s", (10,20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1, cv2.LINE_AA)
            writer.write(frame); i+=1
        cap2.release(); writer.release()

    # Plots
    plt.figure(); plt.plot(times, df["s_px"], label="s (px)"); plt.plot(times, df["s_fit_px"], "--", label="s_fit")
    plt.xlabel("t [s]"); plt.ylabel("s [px]"); plt.legend(); plt.tight_layout()
    plt.savefig(outdir / f"{cfg.output.plots_prefix}_s_t.png", dpi=200); plt.close()

    plt.figure(); plt.plot(times, df["v_along_px_s"], label="v (px/s)")
    plt.xlabel("t [s]"); plt.ylabel("v [px/s]"); plt.legend(); plt.tight_layout()
    plt.savefig(outdir / f"{cfg.output.plots_prefix}_v_t.png", dpi=200); plt.close()

    plt.figure(); plt.plot(times, df["a_along_px_s2"], label="a (px/s^2)")
    plt.xlabel("t [s]"); plt.ylabel("a [px/s^2]"); plt.legend(); plt.tight_layout()
    plt.savefig(outdir / f"{cfg.output.plots_prefix}_a_t.png", dpi=200); plt.close()

    # Fit params sidecar
    with open(outdir / "fit_params.json","w") as f:
        json.dump({"s0_px": (float(s0) if np.isfinite(s0) else None),
                   "v0_px_s": (float(v0) if np.isfinite(v0) else None),
                   "a_px_s2": (float(a) if np.isfinite(a) else None)}, f, indent=2)

    print(f"[OK] CSV: {csv_path}")
    if cfg.output.annotated_video:
        print(f"[OK] Annotated: {outdir / cfg.output.annotated_name}")
    print(f"[OK] Plots saved under: {outdir}")

if __name__ == "__main__":
    main()
