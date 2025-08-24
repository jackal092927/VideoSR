import numpy as np
import pandas as pd
from scipy.signal import savgol_filter


def _interp_nan(series: pd.Series):
    s = series.copy()
    nans = s.isna()
    if nans.all():
        return s.fillna(method=None)
    s_interp = s.interpolate(method="linear", limit_direction="both")
    return s_interp


def smooth_and_derivatives(df: pd.DataFrame, fps: float, cfg: dict) -> pd.DataFrame:
    # Set time vector
    df = df.copy()
    df["t"] = df["frame"] / float(fps)

    # Interpolate missing centroid points
    df["x"] = _interp_nan(df["x"])
    df["y"] = _interp_nan(df["y"])

    # Savitzky–Golay smoothing
    window = int(cfg.get("window", 21))
    if window % 2 == 0:
        window += 1
    poly = int(cfg.get("polyorder", 3))

    x = df["x"].to_numpy()
    y = df["y"].to_numpy()
    t = df["t"].to_numpy()

    # Apply Savitzky-Golay smoothing with proper window length
    if len(x) > poly + 2:
        window_length = min(window, len(x) - (len(x) % 2 == 0))
        polyorder = min(poly, max(1, len(x) // 2 - 1))
        x_s = savgol_filter(x, window_length=window_length, polyorder=polyorder)
    else:
        x_s = x

    if len(y) > poly + 2:
        window_length = min(window, len(y) - (len(y) % 2 == 0))
        polyorder = min(poly, max(1, len(y) // 2 - 1))
        y_s = savgol_filter(y, window_length=window_length, polyorder=polyorder)
    else:
        y_s = y

    # Derivatives (prefer gradient over repeated savgol derivative to reduce lag)
    vx = np.gradient(x_s, t, edge_order=2)
    vy = np.gradient(y_s, t, edge_order=2)
    ax = np.gradient(vx, t, edge_order=2)
    ay = np.gradient(vy, t, edge_order=2)

    out = df.copy()
    out["x"], out["y"] = x_s, y_s
    out["vx"], out["vy"] = vx, vy
    out["ax"], out["ay"] = ax, ay
    return out