from __future__ import annotations

from typing import Dict, Any, Optional, Tuple
from pathlib import Path
import json
import numpy as np
import pandas as pd
import cv2


def _contours_from_mask(mask, min_area: int):
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for c in cnts:
        A = float(cv2.contourArea(c))
        if A < float(min_area):
            continue
        M = cv2.moments(c)
        if M["m00"] == 0:
            continue
        cx = float(M["m10"]/M["m00"]) ; cy = float(M["m01"]/M["m00"]) 
        out.append((cx, cy, A))
    return out


def track_with_custom(video: str, detect_fn, cfg: Dict[str, Any]) -> pd.DataFrame:
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or float(cfg.get('fps_hint', 30.0))
    rows = []
    f = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        pos = detect_fn(frame)
        if pos is None:
            rows.append({"frame": f, "time_s": f/fps, "x_px": np.nan, "y_px": np.nan})
        else:
            cx, cy = pos
            rows.append({"frame": f, "time_s": f/fps, "x_px": float(cx), "y_px": float(cy)})
        f += 1
    cap.release()
    df = pd.DataFrame(rows)
    for c in ["x_px","y_px"]:
        s = df[c].astype(float)
        if not s.isna().all():
            df[c] = s.interpolate(limit_direction='both')
    return df


def track_hsv_nn(video: str, cfg: Dict[str, Any]) -> pd.DataFrame:
    # Defaults aimed at blue; will auto-calibrate if not explicitly provided
    H1_low = int(cfg.get('h_low', 100)); H1_high = int(cfg.get('h_high', 130))
    H2_low = int(cfg.get('h2_low', 168)); H2_high = int(cfg.get('h2_high', 179))
    S_low = int(cfg.get('s_low', 60)); V_low = int(cfg.get('v_low', 60))
    use_red_wrap = bool(cfg.get('use_red_wrap', False))
    open_k = int(cfg.get('open_k', 5)); close_k = int(cfg.get('close_k', 7))
    min_area = int(cfg.get('min_area', 40))
    R = float(cfg.get('search_radius', 60))
    auto_hsv = bool(cfg.get('auto_hsv', True))

    # Allow direct hsv_lower/upper override from plan
    hsv_lower = cfg.get('hsv_lower'); hsv_upper = cfg.get('hsv_upper')
    # Debugging controls
    debug_dir = cfg.get('debug_dir')
    debug_first_n = int(cfg.get('debug_first_n', 10))
    dbg = {
        'explicit_hsv': bool(hsv_lower is not None and hsv_upper is not None),
        'hsv_lower': hsv_lower,
        'hsv_upper': hsv_upper,
        'auto_hsv': auto_hsv,
        'use_red_wrap': use_red_wrap,
        'samples': [],
        'safety': {},
        'frames': [],
    }

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or float(cfg.get('fps_hint', 30.0))
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Auto-calibrate hue window on first few frames if no explicit range provided
    if hsv_lower is None or hsv_upper is None:
        if auto_hsv:
            sample = []
            for _ in range(5):
                ok, frame0 = cap.read()
                if not ok:
                    break
                hsv0 = cv2.cvtColor(frame0, cv2.COLOR_BGR2HSV)
                Hc = hsv0[...,0]; S = hsv0[...,1]; V = hsv0[...,2]
                mask_sv = (S > 50) & (V > 50)
                if np.count_nonzero(mask_sv) > 0:
                    hist = cv2.calcHist([Hc.astype(np.uint8)], [0], mask_sv.astype(np.uint8), [180], [0,180])
                    peak = int(np.argmax(hist))
                    sample.append(peak)
                    if debug_dir and len(dbg['samples']) < debug_first_n:
                        dbg['samples'].append({'peak': int(peak)})
            if sample:
                peak = int(np.median(sample))
                width = 10
                lo = max(0, peak - width); hi = min(179, peak + width)
                H1_low, H1_high = lo, hi
        # reset to frame 0
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    # Optional: small refinement search around provided HSV ranges (deterministic, not auto)
    # Trigger only when explicit ranges exist (from plan/CLI/backfill), auto_hsv is False, and user enables calibrate_refine
    try:
        calibrate_refine = bool(cfg.get('calibrate_refine', False))
        refine_frames = int(cfg.get('refine_frames', 10))
        refine_hue_deg = int(cfg.get('refine_hue_deg', 5))
        refine_v_delta = int(cfg.get('refine_v_delta', 20))
    except Exception:
        calibrate_refine = False
        refine_frames = 10
        refine_hue_deg = 5
        refine_v_delta = 20

    def _eval_candidate(lower_hsv: Tuple[int,int,int], upper_hsv: Tuple[int,int,int]):
        # Evaluate over first N frames; return (valid_count, jitter, mean_circ, mean_area_frac)
        pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        pts = []
        circs = []
        areas = []
        # morphology kernels consistent with main loop
        SEo = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_k, open_k))
        SEc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_k, close_k))
        for i in range(refine_frames):
            ok, fr = cap.read()
            if not ok:
                break
            hsvf = cv2.cvtColor(fr, cv2.COLOR_BGR2HSV)
            ms = cv2.inRange(hsvf, tuple(int(v) for v in lower_hsv), tuple(int(v) for v in upper_hsv))
            ms = cv2.morphologyEx(ms, cv2.MORPH_OPEN, SEo)
            ms = cv2.morphologyEx(ms, cv2.MORPH_CLOSE, SEc)
            cnts, _ = cv2.findContours(ms, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not cnts:
                continue
            c = max(cnts, key=cv2.contourArea)
            A = float(cv2.contourArea(c))
            P = float(cv2.arcLength(c, True)) if len(c) >= 5 else 0.0
            circ = (4.0*np.pi*A/(P*P)) if P > 0 else 0.0
            M = cv2.moments(c)
            if M['m00'] == 0:
                continue
            cx = float(M['m10']/M['m00']); cy = float(M['m01']/M['m00'])
            pts.append((cx, cy))
            circs.append(float(circ))
            areas.append(A)
        # restore position
        cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
        valid = len(pts)
        if valid <= 1:
            jitter = 1e9
        else:
            dsum = 0.0
            for i in range(1, len(pts)):
                dsum += float(np.hypot(pts[i][0]-pts[i-1][0], pts[i][1]-pts[i-1][1]))
            jitter = dsum / (len(pts)-1)
        mean_circ = float(np.mean(circs)) if circs else 0.0
        mean_area_frac = float(np.mean(areas)/(W*H)) if areas else 0.0
        return valid, jitter, mean_circ, mean_area_frac

    if (hsv_lower is not None and hsv_upper is not None) and (not auto_hsv) and calibrate_refine:
        try:
            base_lo = [int(v) for v in hsv_lower]
            base_hi = [int(v) for v in hsv_upper]
            # construct small grid around base hue center and V lower bound
            h_center = int(round((base_lo[0] + base_hi[0]) / 2))
            width = int(max(2, base_hi[0] - base_lo[0]))
            hue_shifts = [-refine_hue_deg, 0, refine_hue_deg]
            v_lows = [base_lo[2], min(255, base_lo[2] + refine_v_delta)]
            candidates = []
            for hs in hue_shifts:
                c = (h_center + hs)
                loH = max(0, c - width//2)
                hiH = min(179, loH + width)
                for v0 in v_lows:
                    lo = [loH, base_lo[1], int(v0)]
                    hi = [hiH, base_hi[1], base_hi[2]]
                    candidates.append((lo, hi))
            # include original as candidate 0
            candidates.insert(0, (base_lo, base_hi))
            scored = []
            for lo, hi in candidates:
                valid, jitter, mean_circ, mean_area_frac = _eval_candidate(lo, hi)
                scored.append({
                    'lo': lo, 'hi': hi,
                    'valid': int(valid), 'jitter': float(jitter),
                    'mean_circ': float(mean_circ), 'mean_area_frac': float(mean_area_frac),
                })
            # sort by: max valid -> min jitter -> max mean_circ -> min area_frac
            scored.sort(key=lambda s: (-s['valid'], s['jitter'], -s['mean_circ'], s['mean_area_frac']))
            best = scored[0]
            hsv_lower = tuple(int(v) for v in best['lo'])
            hsv_upper = tuple(int(v) for v in best['hi'])
            dbg['refine'] = {
                'enabled': True,
                'candidates': scored,
                'chosen': {'hsv_lower': list(hsv_lower), 'hsv_upper': list(hsv_upper)}
            }
        except Exception:
            # if anything fails, keep original and note disabled
            dbg['refine'] = {'enabled': False, 'error': True}

    # Safety pre-check on first few frames
    safety_N = int(cfg.get('safety_frames', 10))
    border_thresh = int(cfg.get('border_thresh_px', 5))
    large_area_ratio = float(cfg.get('large_area_ratio', 0.10))
    border_hits = 0
    large_area_hits = 0
    valid = 0
    for _ in range(safety_N):
        ok, fs = cap.read()
        if not ok:
            break
        hsvs = cv2.cvtColor(fs, cv2.COLOR_BGR2HSV)
        if hsv_lower is not None and hsv_upper is not None:
            lower = tuple(int(v) for v in hsv_lower)
            upper = tuple(int(v) for v in hsv_upper)
            ms = cv2.inRange(hsvs, lower, upper)
        else:
            m1s = cv2.inRange(hsvs, (H1_low, S_low, V_low), (H1_high, 255, 255))
            # only include red wrap if explicitly requested
            if bool(cfg.get('use_red_wrap', False)):
                m2s = cv2.inRange(hsvs, (H2_low, S_low, V_low), (H2_high, 255, 255))
                ms = cv2.bitwise_or(m1s, m2s)
            else:
                ms = m1s
        cnts, _ = cv2.findContours(ms, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            continue
        c = max(cnts, key=cv2.contourArea)
        A = float(cv2.contourArea(c))
        M = cv2.moments(c)
        if M["m00"] == 0:
            continue
        cx = float(M["m10"]/M["m00"]); cy = float(M["m01"]/M["m00"]) 
        valid += 1
        if cx < border_thresh or cy < border_thresh or cx > (W-1-border_thresh) or cy > (H-1-border_thresh):
            border_hits += 1
        if A > large_area_ratio * (W*H):
            large_area_hits += 1
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    dbg['safety'] = {
        'valid_checks': int(valid),
        'border_hits': float(border_hits),
        'large_area_hits': float(large_area_hits),
        'border_ratio': float((border_hits/valid) if valid else 0.0),
        'large_area_ratio_hits': float((large_area_hits/valid) if valid else 0.0),
    }
    if valid > 0 and (border_hits/valid > 0.6 or large_area_hits/valid > 0.3):
        df_fb = track_contour(video, {'threshold_value': int(cfg.get('threshold_value', 127)), 'min_area': min_area, 'fps_hint': fps})
        df_fb.attrs['tracker_info'] = {
            'fallback_used': True,
            'fallback_method': 'contour',
            'border_ratio': border_hits/valid,
            'large_area_ratio_hits': large_area_hits/valid,
        }
        if debug_dir:
            p = Path(debug_dir)
            p.mkdir(parents=True, exist_ok=True)
            (p / 'debug.json').write_text(json.dumps(dbg, indent=2))
        return df_fb
    rows = []
    f = 0
    prev: Optional[Tuple[float,float]] = None
    SEo = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_k, open_k))
    SEc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_k, close_k))
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        if hsv_lower is not None and hsv_upper is not None:
            lower = tuple(int(v) for v in hsv_lower)
            upper = tuple(int(v) for v in hsv_upper)
            mask = cv2.inRange(hsv, lower, upper)
        else:
            m1 = cv2.inRange(hsv, (H1_low, S_low, V_low), (H1_high, 255, 255))
            if bool(cfg.get('use_red_wrap', False)):
                m2 = cv2.inRange(hsv, (H2_low, S_low, V_low), (H2_high, 255, 255))
                mask = cv2.bitwise_or(m1, m2)
            else:
                mask = m1
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, SEo)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, SEc)
        cand = _contours_from_mask(mask, min_area)
        # Optional shape-aware gating (prefer circular blobs)
        chosen_area = np.nan
        confirm_ball = bool(cfg.get('confirm_ball', False))
        if confirm_ball:
            cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            circ_list = []  # (circ, cx, cy, area)
            for c in cnts:
                A = float(cv2.contourArea(c))
                if A < float(min_area):
                    continue
                P = float(cv2.arcLength(c, True)) if len(c) >= 5 else 0.0
                circ = (4.0*np.pi*A/(P*P)) if P > 0 else 0.0
                M = cv2.moments(c)
                if M['m00'] == 0:
                    continue
                cx = float(M['m10']/M['m00']); cy = float(M['m01']/M['m00'])
                circ_list.append((circ, cx, cy, A))
            # Combine circularity with NN gating
            if circ_list:
                if prev is None:
                    circ_list.sort(key=lambda z: (-z[0], -z[3]))  # high circ, then larger area
                    cx, cy, chosen_area = circ_list[0][1], circ_list[0][2], circ_list[0][3]
                else:
                    px, py = prev
                    circ_list.sort(key=lambda z: (np.hypot(z[1]-px, z[2]-py) <= R, -z[0], np.hypot(z[1]-px, z[2]-py)))
                    # First element now favors within radius, higher circularity, then proximity
                    cx, cy, chosen_area = circ_list[0][1], circ_list[0][2], circ_list[0][3]
                rows.append({"frame": f, "time_s": f/fps, "x_px": float(cx), "y_px": float(cy)})
                prev = (cx, cy)
                # Save debug info
                if debug_dir and f < debug_first_n:
                    dbg.setdefault('frames', [])
                    dbg['frames'].append({
                        'frame': int(f),
                        'cand_count': int(len(circ_list)),
                        'chosen_cx': float(cx),
                        'chosen_cy': float(cy),
                        'chosen_area': float(chosen_area),
                        'mask_ones': int(np.count_nonzero(mask)),
                        'confirm_ball': True,
                    })
                f += 1
                continue
        cx = cy = np.nan
        if cand:
            if prev is None:
                cx, cy, chosen_area = max(cand, key=lambda k: k[2])
            else:
                px, py = prev
                d = [(np.hypot(cx-px, cy-py), i) for i, (cx, cy, _) in enumerate(cand)]
                d.sort()
                if d and d[0][0] <= R:
                    cx, cy, chosen_area = cand[d[0][1]]
                else:
                    cx, cy, chosen_area = max(cand, key=lambda k: k[2])
        if np.isfinite(cx) and np.isfinite(cy):
            prev = (cx, cy)
        rows.append({"frame": f, "time_s": f/fps, "x_px": float(cx), "y_px": float(cy)})
        # Save debug artifacts for first N frames
        if debug_dir and f < debug_first_n:
            p = Path(debug_dir); p.mkdir(parents=True, exist_ok=True)
            try:
                cv2.imwrite(str(p / f'mask_{f:03d}.png'), mask)
                cv2.imwrite(str(p / f'frame_{f:03d}.png'), frame)
            except Exception:
                pass
            dbg['frames'].append({
                'frame': int(f),
                'cand_count': int(len(cand)),
                'chosen_cx': float(cx) if np.isfinite(cx) else None,
                'chosen_cy': float(cy) if np.isfinite(cy) else None,
                'chosen_area': float(chosen_area) if np.isfinite(chosen_area) else None,
                'mask_ones': int(np.count_nonzero(mask)),
            })
        f += 1
    cap.release()
    df = pd.DataFrame(rows)
    for c in ["x_px","y_px"]:
        s = df[c].astype(float)
        if not s.isna().all():
            df[c] = s.interpolate(limit_direction='both')
    if debug_dir:
        p = Path(debug_dir)
        try:
            (p / 'debug.json').write_text(json.dumps(dbg, indent=2))
        except Exception:
            pass
    return df


def track_contour(video: str, cfg: Dict[str, Any]) -> pd.DataFrame:
    thresh_val = int(cfg.get('threshold_value', 127))
    min_area = int(cfg.get('min_area', 40))
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or float(cfg.get('fps_hint', 30.0))
    rows = []
    f = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY)
        cand = _contours_from_mask(mask, min_area)
        if cand:
            cx, cy, _ = max(cand, key=lambda k: k[2])
        else:
            cx = cy = np.nan
        rows.append({"frame": f, "time_s": f/fps, "x_px": float(cx), "y_px": float(cy)})
        f += 1
    cap.release()
    df = pd.DataFrame(rows)
    for c in ["x_px","y_px"]:
        s = df[c].astype(float)
        if not s.isna().all():
            df[c] = s.interpolate(limit_direction='both')
    return df
