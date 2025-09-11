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
    }
    if not csv_path.exists():
        return report

    df = pd.read_csv(csv_path)
    # detection rate: rows with finite x,y
    ok = np.isfinite(df.get("x_px", np.nan)) & np.isfinite(df.get("y_px", np.nan))
    report["detection_rate"] = float(np.mean(ok)) if len(df) else float("nan")

    # Prefer y_fit_phys vs y_phys if available, else y_fit_px vs y_px
    y_true = None
    y_pred = None
    if "y_fit_phys" in df.columns and "y_phys" in df.columns:
        y_true = df["y_phys"].to_numpy()
        y_pred = df["y_fit_phys"].to_numpy()
    elif "y_fit_px" in df.columns and "y_px" in df.columns:
        y_true = df["y_px"].to_numpy()
        y_pred = df["y_fit_px"].to_numpy()
    if y_true is not None and y_pred is not None:
        report["r2_primary"] = r2_score(y_true, y_pred)

    # Save quick report duplicate
    (outdir / "quick_report.json").write_text(json.dumps(report, indent=2))
    return report

