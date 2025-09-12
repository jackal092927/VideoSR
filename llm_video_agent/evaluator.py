import json
from pathlib import Path
import numpy as np
import pandas as pd


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if mask.sum() < 3:
        return float("nan")
    y_t = y_true[mask]
    y_p = y_pred[mask]
    ss_res = float(np.sum((y_t - y_p) ** 2))
    ss_tot = float(np.sum((y_t - np.mean(y_t)) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def evaluate_outputs(outdir: Path) -> dict:
    csv_path = outdir / "measurements.csv"
    report = {
        "csv_exists": csv_path.exists(),
        "annotated_exists": (outdir / "annotated.mp4").exists(),
        "plots_dir_exists": (outdir / "plots").exists(),
        "fit_params_exists": (outdir / "fit_params.json").exists(),
        "detection_rate": float("nan"),
        "r2_primary": float("nan"),
        "r2_x": float("nan"),
        "r2_y": float("nan"),
        "r2_primary_axis": None,
        "rmse_x": float("nan"),
        "rmse_y": float("nan"),
        "rmse_2d": float("nan"),
    }
    if not csv_path.exists():
        return report

    df = pd.read_csv(csv_path)
    # detection rate: rows with finite x,y
    ok = np.isfinite(df.get("x_px", np.nan)) & np.isfinite(df.get("y_px", np.nan))
    report["detection_rate"] = float(np.mean(ok)) if len(df) else float("nan")

    # Compute R2 per axis (px space)
    if {'x_px','x_fit_px'}.issubset(df.columns):
        report["r2_x"] = r2_score(df['x_px'].to_numpy(), df['x_fit_px'].to_numpy())
    if {'y_px','y_fit_px'}.issubset(df.columns):
        report["r2_y"] = r2_score(df['y_px'].to_numpy(), df['y_fit_px'].to_numpy())
    # Choose primary axis by variance
    var_x = float(np.nanvar(df.get('x_px', pd.Series(dtype=float)).to_numpy())) if 'x_px' in df.columns else 0.0
    var_y = float(np.nanvar(df.get('y_px', pd.Series(dtype=float)).to_numpy())) if 'y_px' in df.columns else 0.0
    report['r2_primary_axis'] = 'y' if var_y >= var_x else 'x'
    if report['r2_primary_axis'] == 'y':
        report['r2_primary'] = report['r2_y']
    else:
        report['r2_primary'] = report['r2_x']

    # Save quick report duplicate
    (outdir / "quick_report.json").write_text(json.dumps(report, indent=2))

    # RMSE vs GT (optional): look for gt.csv or *_gt.csv
    try:
        gt_path = outdir / 'gt.csv'
        if not gt_path.exists():
            import glob
            matches = glob.glob(str(outdir / '*_gt.csv'))
            if matches:
                gt_path = Path(matches[0])
        if gt_path.exists():
            gt = pd.read_csv(gt_path)
            # Align by frame if possible
            if 'frame' in gt.columns and 'frame' in df.columns:
                merged = pd.merge(df[['frame','x_px','y_px']], gt[['frame','x','y']], on='frame', how='inner')
                mx = merged['x_px'].to_numpy(); my = merged['y_px'].to_numpy()
                gx = merged['x'].to_numpy(); gy = merged['y'].to_numpy()
            else:
                n = min(len(df), len(gt))
                mx = df['x_px'].to_numpy()[:n]; my = df['y_px'].to_numpy()[:n]
                gx = gt['x'].to_numpy()[:n]; gy = gt['y'].to_numpy()[:n]
            mask = np.isfinite(mx) & np.isfinite(my) & np.isfinite(gx) & np.isfinite(gy)
            if mask.sum() > 0:
                dx = mx[mask] - gx[mask]; dy = my[mask] - gy[mask]
                report['rmse_x'] = float(np.sqrt(np.mean(dx*dx)))
                report['rmse_y'] = float(np.sqrt(np.mean(dy*dy)))
                report['rmse_2d'] = float(np.sqrt(np.mean(dx*dx + dy*dy)))
    except Exception:
        pass

    # Update quick report with RMSE if computed
    (outdir / "quick_report.json").write_text(json.dumps(report, indent=2))
    return report
    return report
