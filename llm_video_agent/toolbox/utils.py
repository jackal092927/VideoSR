from __future__ import annotations

from typing import Dict, Any
import numpy as np
import pandas as pd


def make_interior_mask(df: pd.DataFrame, radius: int = 12, pad: int = 5) -> np.ndarray:
    # For now, use image bounds unknown; mask based on percentiles of observed positions
    x = df['x_px'].to_numpy(); y = df['y_px'].to_numpy()
    m = np.isfinite(x) & np.isfinite(y)
    if not m.any():
        return np.zeros(len(df), dtype=bool)
    x_valid = x[m]; y_valid = y[m]
    xmin, xmax = np.percentile(x_valid, [1, 99])
    ymin, ymax = np.percentile(y_valid, [1, 99])
    return (x > xmin + pad) & (x < xmax - pad) & (y > ymin + pad) & (y < ymax - pad)

