#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
analyze_incline.py — Track a toy car on an inclined plane and extract physics
-----------------------------------------------------------------------------
(Shortened header in this patch cell; the functionality remains identical.)
"""

import argparse, json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List, Tuple, Dict

import cv2, numpy as np, pandas as pd

try:
    from scipy.signal import savgol_filter
except Exception:
    savgol_filter = None

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

@dataclass
class InclineConfig:
    auto_detect: bool = True
    manual_points: Optional[List[int]] = None
    band_halfwidth_px: int = 25
    hough_canny1: int = 80
    hough_canny2: int = 150
    hough_threshold: int = 80
    hough_min_line_len: int = 100
    hough_max_line_gap: int = 10
    min_angle_deg: float = 10.0
    max_angle_deg: float = 80.0

@dataclass
class ForegroundConfig:
    bg_samples: int = 60
    bg_stride: int = 5
    diff_thresh: int = 30
    morph_kernel: int = 5
    min_area: int = 120

@dataclass
class ColorConfig:
    enable_include: bool = False
    include_hsv_lower: Optional[List[int]] = None
    include_hsv_upper: Optional[List[int]] = None
    enable_exclude_skin: bool = True
    skin_lower: List[int] = field(default_factory=lambda: [0,40,60])
    skin_upper: List[int] = field(default_factory=lambda: [25,180,255])

@dataclass
class TrackingConfig:
    prefer_monotonic_downhill: bool = True
    gate_px: float = 60.0
    smooth_method: str = 'savgol'
    smooth_window: int = 9
    smooth_poly: int = 2

@dataclass
class PhysicsConfig:
    pixels_per_meter: Optional[float] = None
    y_axis_down: bool = True

@dataclass
class OutputConfig:
    annotated_video: bool = True
    csv_name: str = 'incline_measurements.csv'
    annotated_name: str = 'incline_annotated.mp4'
    plots_prefix: str = 'incline_plot'

@dataclass
class AppConfig:
    incline: InclineConfig = field(default_factory=InclineConfig)
    fg: ForegroundConfig = field(default_factory=ForegroundConfig)
    color: ColorConfig = field(default_factory=ColorConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    physics: PhysicsConfig = field(default_factory=PhysicsConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

def load_config(path: Optional[Path]) -> AppConfig:
    if path is None:
        return AppConfig()
    with open(path, 'r') as f:
        raw = json.load(f)
    def dc(dc_cls, d):
        return dc_cls(**{k: d.get(k, getattr(dc_cls(), k)) for k in dc_cls().__dict__.keys()})
    return AppConfig(
        incline=dc(InclineConfig, raw.get('incline', {})),
        fg=dc(ForegroundConfig, raw.get('fg', {})),
        color=dc(ColorConfig, raw.get('color', {})),
        tracking=dc(TrackingConfig, raw.get('tracking', {})),
        physics=dc(PhysicsConfig, raw.get('physics', {})),
        output=dc(OutputConfig, raw.get('output', {})),
    )

def load_video(path: Path):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise IOError(f'Cannot open video: {path}')
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    N = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    return cap, float(fps), (W,H), N

def sample_background(video_path: Path, fps: float, N: int, cfg: ForegroundConfig):
    cap = cv2.VideoCapture(str(video_path))
    frames = []; stride = max(1, int(cfg.bg_stride))
    step = max(1, int(N / max(1, cfg.bg_samples*stride)))
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret: break
        if (idx % (stride*step)) == 0:
            frames.append(frame)
            if len(frames) >= cfg.bg_samples: break
        idx += 1
    cap.release()
    if not frames: raise RuntimeError('No frames sampled for background')
    return np.median(np.stack(frames, axis=0), axis=0).astype(np.uint8)

def detect_incline_line(image_bgr, inc: InclineConfig):
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, inc.hough_canny1, inc.hough_canny2)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=inc.hough_threshold,
                            minLineLength=inc.hough_min_line_len, maxLineGap=inc.hough_max_line_gap)
    best=None
    if lines is not None:
        for x1,y1,x2,y2 in lines[:,0,:]:
            dx,dy = x2-x1, y2-y1
            angle = abs(np.degrees(np.arctan2(dy,dx))); angle = 180-angle if angle>90 else angle
            if angle < inc.min_angle_deg or angle > inc.max_angle_deg: continue
            length = (dx*dx+dy*dy)**0.5
            if best is None or length>best[0]:
                best=(length,(x1,y1,x2,y2))
    if best is None: raise RuntimeError('Could not detect incline; set manual_points.')
    x1,y1,x2,y2 = best[1]
    v = np.array([x2-x1, y2-y1], float); n = np.linalg.norm(v)
    if n<1e-6: raise RuntimeError('Degenerate incline line')
    u = v/n; x0 = np.array([x1,y1], float)
    return u, x0, (x1,y1,x2,y2)

def band_mask(shape, u, x0, halfwidth):
    H,W = shape[:2]
    yy,xx = np.mgrid[0:H,0:W]
    P = np.stack([xx,yy],axis=-1).astype(float)
    n = np.array([-u[1], u[0]])
    dist = np.abs((P - x0) @ n)
    return dist <= halfwidth

def hsv_mask(frame_bgr, color_cfg):
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    include = np.ones(hsv.shape[:2], dtype=np.uint8)*255
    if color_cfg.enable_include and color_cfg.include_hsv_lower and color_cfg.include_hsv_upper:
        include = cv2.inRange(hsv, np.array(color_cfg.include_hsv_lower,np.uint8),
                                   np.array(color_cfg.include_hsv_upper,np.uint8))
    exclude = np.zeros(hsv.shape[:2], dtype=np.uint8)
    if color_cfg.enable_exclude_skin and color_cfg.skin_lower and color_cfg.skin_upper:
        exclude = cv2.inRange(hsv, np.array(color_cfg.skin_lower,np.uint8),
                                   np.array(color_cfg.skin_upper,np.uint8))
    return include, exclude

def detect_candidates(frame_bgr, bg_bgr, band, fg_cfg, color_cfg):
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    bg_gray = cv2.cvtColor(bg_bgr, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(gray, bg_gray)
    _, mask = cv2.threshold(diff, fg_cfg.diff_thresh, 255, cv2.THRESH_BINARY)
    include, exclude = hsv_mask(frame_bgr, color_cfg)
    mask = cv2.bitwise_and(mask, include)
    mask = cv2.bitwise_and(mask, cv2.bitwise_not(exclude))
    mask = cv2.bitwise_and(mask, (band.astype(np.uint8))*255)
    if fg_cfg.morph_kernel>1:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (fg_cfg.morph_kernel, fg_cfg.morph_kernel))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    cnts,_ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cands = []
    for c in cnts:
        area = cv2.contourArea(c)
        if area < fg_cfg.min_area: continue
        x,y,w,h = cv2.boundingRect(c)
        M = cv2.moments(c)
        if M['m00']<=0: continue
        cx = M['m10']/M['m00']; cy = M['m01']/M['m00']
        cands.append({'centroid':(cx,cy),'bbox':(x,y,w,h),'area':area})
    return cands, mask

def project_point_on_line(pt, u, x0):
    s = (pt - x0) @ u
    q = x0 + s*u
    return float(s), q

def smooth_series(arr, method, window, poly):
    if method=='none': return arr.copy()
    if method=='savgol' and savgol_filter is not None and len(arr)>=max(window,5):
        w = window if window%2==1 else window+1
        w = min(w, len(arr)-(1-len(arr)%2)) if len(arr)>2 else 3
        w = max(5,w); p = min(poly, w-1)
        try: return savgol_filter(arr, window_length=w, polyorder=p, mode='interp')
        except Exception: return arr.copy()
    return arr.copy()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./incline_config.json')
    parser.add_argument('--video', type=str, default='./2419_1744339511.mp4')
    parser.add_argument('--outdir', type=str, default='incline_out')
    args = parser.parse_args()

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    cfg = load_config(Path(args.config) if args.config else None)

    cap, fps, (W,H), N = load_video(Path(args.video))
    bg = sample_background(Path(args.video), fps, N, cfg.fg)

    if cfg.incline.auto_detect:
        u, x0, seg = detect_incline_line(bg, cfg.incline)
        # orient from top to bottom (larger y is lower if y-axis down)
        x1,y1,x2,y2 = seg
        if y2 < y1:
            x0 = np.array([x2,y2], float)
            u  = -u
        else:
            x0 = np.array([x1,y1], float)
    else:
        if not cfg.incline.manual_points or len(cfg.incline.manual_points)!=4:
            raise RuntimeError('manual_points must be [x1,y1,x2,y2]')
        x1,y1,x2,y2 = cfg.incline.manual_points
        v = np.array([x2-x1, y2-y1], float); u = v/(np.linalg.norm(v)+1e-9); x0 = np.array([x1,y1], float)

    band = band_mask((H,W), u, x0, cfg.incline.band_halfwidth_px)

    writer=None
    if cfg.output.annotated_video:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(str(outdir/cfg.output.annotated_name), fourcc, fps, (W,H))

    prev_pt=None; prev_s=None; alpha_mono=30.0
    rows=[]; fidx=0
    while True:
        ret, frame = cap.read()
        if not ret: break
        t = fidx/fps
        cands,_ = detect_candidates(frame, bg, band, cfg.fg, cfg.color)

        chosen=None
        if cands:
            scores=[]
            for cand in cands:
                cx,cy = cand['centroid']; pt=np.array([cx,cy], float)
                s,_ = project_point_on_line(pt, u, x0)
                cost=0.0
                if prev_pt is not None:
                    d = np.linalg.norm(pt - prev_pt); cost += min(d, cfg.tracking.gate_px)
                    if cfg.tracking.prefer_monotonic_downhill and prev_s is not None:
                        delta = s - prev_s; cost += (-alpha_mono)*max(0.0, delta)
                else:
                    cost += s
                scores.append((cost, cand, s, pt))
            scores.sort(key=lambda z:z[0])
            if prev_pt is not None:
                if scores[0][0] < cfg.tracking.gate_px*1.5:
                    chosen = (scores[0][1], scores[0][2], scores[0][3])
            else:
                chosen = (scores[0][1], scores[0][2], scores[0][3])

        if chosen is not None:
            cand,s,pt = chosen
            prev_pt=pt; prev_s=s
            rows.append({'frame':fidx,'time_s':t,'x':float(pt[0]),'y':float(pt[1]),'s':float(s)})
            if writer is not None:
                overlay=frame.copy()
                band_u8=(band.astype(np.uint8)*255)
                band_col=cv2.cvtColor(band_u8, cv2.COLOR_GRAY2BGR); band_col[:,:,1]=np.maximum(band_col[:,:,1], band_u8)
                frame=cv2.addWeighted(overlay,0.7, band_col,0.3,0)
                p0=tuple(np.int32(x0)); p1=tuple(np.int32(x0 + u*2000)); cv2.line(frame, p0, p1, (0,255,0),2)
                cv2.circle(frame,(int(pt[0]),int(pt[1])),6,(0,0,255),-1)
                cv2.putText(frame, f's={s:.1f}px', (int(pt[0])+8,int(pt[1])-8), cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1,cv2.LINE_AA)
        else:
            rows.append({'frame':fidx,'time_s':t,'x':np.nan,'y':np.nan,'s':np.nan})
            if writer is not None:
                overlay=frame.copy(); band_u8=(band.astype(np.uint8)*255)
                band_col=cv2.cvtColor(band_u8, cv2.COLOR_GRAY2BGR); band_col[:,:,1]=np.maximum(band_col[:,:,1], band_u8)
                frame=cv2.addWeighted(overlay,0.7, band_col,0.3,0)
                p0=tuple(np.int32(x0)); p1=tuple(np.int32(x0 + u*2000)); cv2.line(frame, p0, p1, (0,255,0),2)

        if writer is not None:
            cv2.putText(frame, f'frame={fidx} t={t:.2f}s', (10,20), cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1,cv2.LINE_AA)
            writer.write(frame)

        fidx+=1

    cap.release(); 
    if writer is not None: writer.release()

    df=pd.DataFrame(rows).sort_values('frame').reset_index(drop=True)
    s=df['s'].to_numpy(float)
    isnan=np.isnan(s); 
    if isnan.any():
        idx=np.arange(len(s)); valid=~isnan
        if valid.sum()>=2:
            interp=np.interp(idx, idx[valid], s[valid])
            run=None
            for i,nan in enumerate(isnan):
                if nan and run is None: run=i
                if not nan and run is not None:
                    if (i-1-run+1)<=8: s[run:i]=interp[run:i]
                    run=None
            if run is not None:
                i = len(s)
                if (i-1-run+1) <= 8:
                    s[run:i] = interp[run:i]
    df['s_interp']=s

    def smooth(arr):
        if cfg.tracking.smooth_method=='savgol' and savgol_filter is not None and len(arr)>=9:
            return savgol_filter(arr, window_length=9, polyorder=2, mode='interp')
        return arr

    s_smooth=smooth(df['s_interp'].to_numpy(float))
    t=df['time_s'].to_numpy(float)
    vs=np.gradient(s_smooth,t,edge_order=2)
    acc=np.gradient(vs,t,edge_order=2)

    mask=np.isfinite(s_smooth)&np.isfinite(t)
    if mask.sum()>=5:
        A=np.vstack([np.ones_like(t[mask]), t[mask], 0.5*(t[mask]**2)]).T
        sol,_res,_r,_s=np.linalg.lstsq(A, s_smooth[mask], rcond=None)
        s0,v0,a=sol; s_fit=s0+v0*t+0.5*a*(t**2)
    else:
        s0=v0=a=np.nan; s_fit=np.full_like(t, np.nan)

    out=pd.DataFrame({
        'frame':df['frame'],'time_s':t,'x_px':df['x'],'y_px':df['y'],
        's_px':df['s_interp'],'s_smooth_px':s_smooth,'v_along_px_s':vs,'a_along_px_s2':acc,'s_fit_px':s_fit
    })

    csv_path=Path(args.outdir)/cfg.output.csv_name
    out.to_csv(csv_path, index=False)

    plt.figure(); plt.plot(t,out['s_px'],label='s (px)'); plt.plot(t,out['s_fit_px'],'--',label='s_fit')
    plt.xlabel('t [s]'); plt.ylabel('s [px]'); plt.legend(); plt.tight_layout()
    plt.savefig(Path(args.outdir)/f"{cfg.output.plots_prefix}_s_t.png", dpi=200); plt.close()

    plt.figure(); plt.plot(t,out['v_along_px_s'],label='v (px/s)')
    plt.xlabel('t [s]'); plt.ylabel('v [px/s]'); plt.legend(); plt.tight_layout()
    plt.savefig(Path(args.outdir)/f"{cfg.output.plots_prefix}_v_t.png", dpi=200); plt.close()

    plt.figure(); plt.plot(t,out['a_along_px_s2'],label='a (px/s^2)')
    plt.xlabel('t [s]'); plt.ylabel('a [px/s^2]'); plt.legend(); plt.tight_layout()
    plt.savefig(Path(args.outdir)/f"{cfg.output.plots_prefix}_a_t.png", dpi=200); plt.close()

    with open(Path(args.outdir)/'fit_params.json','w') as f:
        json.dump({'s0_px':float(s0) if np.isfinite(s0) else None,
                   'v0_px':float(v0) if np.isfinite(v0) else None,
                   'a_px_s2':float(a) if np.isfinite(a) else None}, f, indent=2)

    print(f'[OK] CSV: {csv_path}')
    if cfg.output.annotated_video: print(f'[OK] Annotated: {Path(args.outdir)/cfg.output.annotated_name}')
    print(f'[OK] Plots saved to {args.outdir}')

if __name__=='__main__':
    main()
