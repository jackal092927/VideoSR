from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any

from ..planner.snippet_safe import validate_snippet


ANALYZER_TEMPLATE = """#!/usr/bin/env python3
import argparse, json, sys
from pathlib import Path
import numpy as np, pandas as pd
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Ensure repo root is on sys.path for llm_video_agent imports
def _add_repo_root():
    here = Path(__file__).resolve()
    for p in here.parents:
        if (p / 'llm_video_agent').exists():
            sys.path.insert(0, str(p))
            return
_add_repo_root()

from llm_video_agent.toolbox import tracking as T
from llm_video_agent.toolbox import denoise as D
from llm_video_agent.toolbox import fitting as F
from llm_video_agent.toolbox import overlay as O
from llm_video_agent.toolbox import utils as U

# Optional injected snippets
HAS_DETECT_SNIPPET = __HAS_DETECT__
HAS_SMOOTH_SNIPPET = __HAS_SMOOTH__
HAS_FIT_SNIPPET = __HAS_FIT__

# Stub to be optionally replaced by snippet
def detect_custom(frame_bgr):
    return None

def smooth_kalman_custom(df, cfg):
    return None

def fit_motion_custom(t, x_s, y_s, motion_class, cfg):
    return None, None, None

__INJECT_DETECT__
__INJECT_SMOOTH__
__INJECT_FIT__

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--video', required=True)
    ap.add_argument('--outdir', required=True)
    args = ap.parse_args()
    video = args.video
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    plots_dir = outdir / 'plots'; plots_dir.mkdir(exist_ok=True)

    # Locate plan next to this script
    plan_path = Path(__file__).with_name('plan.normalized.json')
    if not plan_path.exists():
        plan_path_hint = Path(__file__).with_name('plan.path')
        if plan_path_hint.exists():
            hint = plan_path_hint.read_text().strip()
            plan_path = Path(hint)
    plan = json.loads(Path(plan_path).read_text())
    tracker = plan['tracker']; denoiser = plan['denoiser']; fitter = plan['fitter']
    masks = plan.get('masks', {}); overlays = plan.get('overlays', {})

    # 1) Track
    if HAS_DETECT_SNIPPET:
        params = dict(tracker.get('params', {}))
        debug_dir = outdir / '_debug'
        debug_dir.mkdir(exist_ok=True)
        params['debug_dir'] = str(debug_dir)
        params['debug_first_n'] = int(params.get('debug_first_n', 10))
        (outdir / 'tracker_used_params.json').write_text(json.dumps(params, indent=2))
        df = T.track_with_custom(video, detect_custom, params)
    else:
        name = tracker.get('name', 'hsv_nn')
        params = dict(tracker.get('params', {}))
        debug_dir = outdir / '_debug'
        debug_dir.mkdir(exist_ok=True)
        params['debug_dir'] = str(debug_dir)
        params['debug_first_n'] = int(params.get('debug_first_n', 10))
        (outdir / 'tracker_used_params.json').write_text(json.dumps(params, indent=2))
        if name == 'hsv_nn':
            df = T.track_hsv_nn(video, params)
        elif name == 'contour':
            df = T.track_contour(video, params)
        else:
            # fall back to hsv
            df = T.track_hsv_nn(video, params)

    # Persist tracker diagnostics if present
    try:
        info = getattr(df, 'attrs', {}).get('tracker_info', {})
        if isinstance(info, dict) and info:
            (outdir / 'tracker_info.json').write_text(json.dumps(info, indent=2))
    except Exception:
        pass

    # Save raw measurements right after tracking (pre-denoise)
    try:
        df_raw = df.copy()
        df_raw.to_csv(outdir / 'measurements_raw.csv', index=False)
    except Exception:
        pass

    # Border mask (for training subset) and occlusion strategy can be applied later
    border = (masks.get('border') or {})
    train_mask = U.make_interior_mask(df, radius=border.get('radius', 12), pad=border.get('pad', 5))

    # 2) Denoise
    dname = denoiser.get('name', 'savgol')
    if HAS_SMOOTH_SNIPPET:
        df_s = smooth_kalman_custom(df, denoiser.get('params', {}))
        if df_s is None:
            dname = dname
        else:
            df = df_s
            dname = 'custom'
    if dname == 'savgol':
        df = D.smooth_savgol(df, denoiser.get('params', {}))
    elif dname == 'kalman':
        df = D.smooth_kalman(df, denoiser.get('params', {}))

    # Save measurements after denoise
    try:
        df_den = df.copy()
        df_den.to_csv(outdir / 'measurements_denoised.csv', index=False)
    except Exception:
        pass

    # 3) Fit candidates and pick best
    t = df['time_s'].to_numpy(); x = df['x_px'].to_numpy(); y = df['y_px'].to_numpy()
    results = {}
    for cand in fitter.get('candidates', ['projectile','linear']):
        cand = str(cand)
        if HAS_FIT_SNIPPET and cand in ('projectile','linear'):
            params, xfit, yfit = fit_motion_custom(t, x, y, cand, fitter)
            if xfit is not None and yfit is not None:
                results[cand] = {'params': params, 'x_fit': np.asarray(xfit), 'y_fit': np.asarray(yfit)}
                continue
        if cand == 'projectile':
            results[cand] = F.fit_projectile_xy(t, x, y, fitter)
        elif cand == 'linear':
            results[cand] = F.fit_linear_xy(t, x, y, fitter)
        elif cand == 'free_fall':
            results[cand] = F.fit_free_fall(t, x, y, fitter)
        elif cand == 'uniform_accel_1d':
            results[cand] = F.fit_uniform_accel_1d(t, x, y, fitter)
        elif cand == 'sho':
            results[cand] = F.fit_sho_xy(t, x, y, fitter)

    # Pick best by MSE along the dominant axis (higher variance)
    def mse(a, b):
        m = np.isfinite(a) & np.isfinite(b)
        if not m.any(): return 1e9
        return float(np.mean((a[m]-b[m])**2))
    pick_by = fitter.get('pick_by','AIC')
    var_x = np.nanvar(x); var_y = np.nanvar(y)
    primary_axis = 'y' if var_y >= var_x else 'x'
    best_name = None; best_score = 1e99
    for name, res in results.items():
        if primary_axis == 'y':
            score = mse(y, res['y_fit'])
        else:
            score = mse(x, res['x_fit'])
        if score < best_score:
            best_name, best_score = name, score
    if not results:
        # Safety fallback: ensure we always have a result
        results['linear_fallback'] = F.fit_linear_xy(t, x, y, fitter)
        best_name = 'linear_fallback'
    best = results.get(best_name, next(iter(results.values())))

    # 4) Save outputs
    out = df.copy()
    out['x_fit_px'] = best['x_fit']
    out['y_fit_px'] = best['y_fit']
    out.to_csv(outdir / 'measurements.csv', index=False)

    # Stage statistics for debugging
    try:
        stats = {}
        if 'df_raw' in locals():
            stats['raw'] = {
                'min_x': float(df_raw['x_px'].min()), 'max_x': float(df_raw['x_px'].max()),
                'min_y': float(df_raw['y_px'].min()), 'max_y': float(df_raw['y_px'].max()),
            }
        if 'df_den' in locals():
            stats['denoised'] = {
                'min_x': float(df_den['x_px'].min()), 'max_x': float(df_den['x_px'].max()),
                'min_y': float(df_den['y_px'].min()), 'max_y': float(df_den['y_px'].max()),
            }
        stats['final'] = {
            'min_x': float(out['x_px'].min()), 'max_x': float(out['x_px'].max()),
            'min_y': float(out['y_px'].min()), 'max_y': float(out['y_px'].max()),
        }
        (outdir / '_debug' / 'stage_stats.json').write_text(json.dumps(stats, indent=2))
    except Exception:
        pass

    (outdir / 'fit_params.json').write_text(json.dumps({'model': best_name, 'params': best.get('params', {})}, indent=2))

    # Plots
    plt.figure(); plt.plot(t, out['x_px'], label='x'); plt.plot(t, out['x_fit_px'], '--', label='x_fit'); plt.legend(); plt.tight_layout(); plt.savefig(plots_dir / 'x_t.png', dpi=150); plt.close()
    plt.figure(); plt.plot(t, out['y_px'], label='y'); plt.plot(t, out['y_fit_px'], '--', label='y_fit'); plt.legend(); plt.tight_layout(); plt.savefig(plots_dir / 'y_t.png', dpi=150); plt.close()

    # Overlay
    O.render_overlay(video, out, outdir / 'annotated.mp4', overlays)

if __name__ == '__main__':
    main()
"""


@dataclass
class HybridCodeGenerator:
    def generate(self, video_path: Path, prompt: str, gen_dir: Path, options: Dict[str, Any] | None = None) -> Path:
        options = options or {}
        # Load normalized plan generated by planner
        plan_path = gen_dir / 'plan.normalized.json'
        if not plan_path.exists():
            # fall back to simple: raise to allow controller to catch? for now create minimal plan
            minimal = {
                'plan_id': 'fallback',
                'tracker': {'name': 'hsv_nn', 'params': {}},
                'denoiser': {'name': 'savgol', 'params': {}},
                'fitter': {'candidates': ['projectile','linear'], 'pick_by': 'MSE', 'robust':'none', 'delta':3.0, 'interior_mask': {'radius':12,'pad':5}},
                'masks': {'border': {'radius':12,'pad':5}},
                'overlays': {'fps':8,'show_smoothed':True,'show_fit':True},
                'snippets': [],
            }
            (gen_dir / 'plan.normalized.json').write_text(json.dumps(minimal, indent=2))
            plan_path = gen_dir / 'plan.normalized.json'

        # Apply CLI overrides (e.g., HSV bounds) to the plan to guarantee correct detection
        try:
            plan_obj = json.loads(plan_path.read_text())
        except Exception:
            plan_obj = None
        if isinstance(plan_obj, dict):
            tracker = plan_obj.get('tracker') or {}
            params = tracker.get('params') or {}
            # detector override
            det_override = options.get('detector')
            if det_override:
                tracker['name'] = det_override
            # hsv overrides
            if options.get('hsv_lower') and options.get('hsv_upper'):
                params['hsv_lower'] = [int(v) for v in options['hsv_lower']]
                params['hsv_upper'] = [int(v) for v in options['hsv_upper']]
                params['auto_hsv'] = False
                tracker['name'] = 'hsv_nn'
            # auto-HSV toggle
            if options.get('auto_hsv'):
                params['auto_hsv'] = True
                params.pop('hsv_lower', None)
                params.pop('hsv_upper', None)

            # Backfill HSV from prompt if auto_hsv is False/absent and plan lacks explicit ranges
            def _infer_hsv_from_prompt(ptext: str):
                p = (ptext or '').lower()
                palette = {
                    'blue': ([100, 80, 80], [130, 255, 255]),
                    'red': ([0, 120, 120], [10, 255, 255]),  # wrap disabled by default
                    'green': ([40, 40, 40], [85, 255, 255]),
                    'yellow': ([20, 80, 80], [35, 255, 255]),
                }
                for color, (lo, hi) in palette.items():
                    if color in p:
                        return lo, hi, color
                # default to blue if no color word found
                return palette['blue'][0], palette['blue'][1], 'blue'

            auto_hsv_flag = bool(params.get('auto_hsv', False))
            has_explicit = ('hsv_lower' in params and 'hsv_upper' in params)
            if not auto_hsv_flag and not has_explicit:
                lo, hi, color_name = _infer_hsv_from_prompt(prompt)
                params['hsv_lower'] = [int(v) for v in lo]
                params['hsv_upper'] = [int(v) for v in hi]
                params['use_red_wrap'] = bool(params.get('use_red_wrap', False)) if color_name == 'red' else False
                params['auto_hsv'] = False
                tracker['name'] = 'hsv_nn'

            tracker['params'] = params
            plan_obj['tracker'] = tracker
            # Denoiser override from CLI
            denoiser_choice = options.get('denoiser')
            if denoiser_choice:
                den = plan_obj.get('denoiser') or {}
                den['name'] = denoiser_choice
                if denoiser_choice == 'savgol':
                    den['params'] = den.get('params') or {"window": 9, "poly": 2}
                elif denoiser_choice == 'kalman':
                    den['params'] = den.get('params') or {"q": 25.0, "r": 6.0}
                else:
                    den['params'] = {}
                plan_obj['denoiser'] = den
            # Thread refinement and shape-aware toggles from CLI into tracker params
            for k in ('calibrate_refine','refine_hue_deg','refine_v_delta','refine_frames','confirm_ball'):
                if k in options and options[k] is not None:
                    params[k] = options[k]
            # Overlay toggles/defaults
            overlays = plan_obj.get('overlays') or {}
            if options.get('overlay_no_fit'):
                overlays['show_fit'] = False
            # Default to not showing fit in hybrid unless explicitly enabled
            if 'show_fit' not in overlays:
                overlays['show_fit'] = False
            plan_obj['overlays'] = overlays
            plan_path.write_text(json.dumps(plan_obj, indent=2))

        # Read optional snippets and validate
        inject_detect = ''
        has_detect = False
        sd = gen_dir / 'snippet_detector.py'
        if sd.exists():
            code = sd.read_text()
            ok, msg = validate_snippet(code, max_lines=80)
            if ok:
                inject_detect = code
                has_detect = True

        inject_smooth = ''
        has_smooth = False
        ss = gen_dir / 'snippet_smooth.py'
        if ss.exists():
            code = ss.read_text()
            ok, msg = validate_snippet(code, max_lines=80)
            if ok:
                inject_smooth = code
                has_smooth = True

        inject_fit = ''
        has_fit = False
        sf = gen_dir / 'snippet_fit.py'
        if sf.exists():
            code = sf.read_text()
            ok, msg = validate_snippet(code, max_lines=80)
            if ok:
                inject_fit = code
                has_fit = True

        content = ANALYZER_TEMPLATE
        content = content.replace('__HAS_DETECT__', 'True' if has_detect else 'False')
        content = content.replace('__HAS_SMOOTH__', 'True' if has_smooth else 'False')
        content = content.replace('__HAS_FIT__', 'True' if has_fit else 'False')
        content = content.replace('__INJECT_DETECT__', inject_detect)
        content = content.replace('__INJECT_SMOOTH__', inject_smooth)
        content = content.replace('__INJECT_FIT__', inject_fit)

        out_path = gen_dir / 'analyzer.py'
        out_path.write_text(content)
        # Also write a sidecar that points to the plan path (runner passes it)
        (gen_dir / 'plan.path').write_text(str(plan_path))
        return out_path
