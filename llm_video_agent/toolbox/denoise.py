from __future__ import annotations

from typing import Dict, Any
import numpy as np
import pandas as pd

try:
    from scipy.signal import savgol_filter
except Exception:
    savgol_filter = None


def smooth_savgol(df: pd.DataFrame, cfg: Dict[str, Any]) -> pd.DataFrame:
    window = int(cfg.get('window', 9))
    poly = int(cfg.get('poly', 2))
    if window % 2 == 0:
        window += 1
    out = df.copy()
    for col in ['x_px','y_px']:
        a = out[col].to_numpy(dtype=float)
        if savgol_filter is not None and len(a) >= max(5, poly+2) and window <= len(a):
            out[col] = savgol_filter(a, window_length=min(window, len(a)-(len(a)%2==0)), polyorder=min(poly, max(1, len(a)//2-1)))
        else:
            # moving average fallback
            w = max(3, window if window % 2 == 1 else window+1)
            if len(a) >= w:
                k = np.ones(w)/w
                out[col] = np.convolve(a, k, mode='same')
    return out


def smooth_kalman(df: pd.DataFrame, cfg: Dict[str, Any]) -> pd.DataFrame:
    # Accept both q/r and noise_cov/measurement_cov keys
    q = float(cfg.get('q', cfg.get('noise_cov', 25.0)))
    r = float(cfg.get('r', cfg.get('measurement_cov', 6.0)))
    t = df['time_s'].to_numpy(); x = df['x_px'].to_numpy(); y = df['y_px'].to_numpy()
    if len(t) < 3:
        return df.copy()
    dt = float(np.median(np.diff(t))) if np.isfinite(np.diff(t)).all() else 1.0
    F = np.array([[1,0,dt,0],[0,1,0,dt],[0,0,1,0],[0,0,0,1]], float)
    H = np.array([[1,0,0,0],[0,1,0,0]], float)
    q11=(dt**4)/4.0; q13=(dt**3)/2.0; q33=(dt**2)
    Q=q*np.array([[q11,0,q13,0],[0,q11,0,q13],[q13,0,q33,0],[0,q13,0,q33]], float)
    R=r*np.eye(2)
    n=len(t)
    X=np.zeros((4,n)); P=np.zeros((4,4,n))
    xp=np.array([x[0], y[0], 0, 0], float); Pp=np.eye(4)*1e3
    for k in range(n):
        xpred=F@xp; Ppred=F@Pp@F.T+Q
        z=np.array([x[k], y[k]]); yk=z-H@xpred
        S=H@Ppred@H.T+R
        K=Ppred@H.T@np.linalg.inv(S)
        xup=xpred+K@yk; Pup=(np.eye(4)-K@H)@Ppred
        X[:,k]=xup; P[:,:,k]=Pup; xp, Pp = xup, Pup
    xs=np.zeros(n); ys=np.zeros(n); xs[-1]=X[0,-1]; ys[-1]=X[1,-1]
    for k in range(n-2, -1, -1):
        C=P[:,:,k]@F.T@np.linalg.inv(F@P[:,:,k]@F.T+Q)
        sm=X[:,k]+C@(np.array([xs[k+1], ys[k+1], 0, 0]) - F@X[:,k])
        xs[k], ys[k] = sm[0], sm[1]
    out = df.copy(); out['x_px'], out['y_px'] = xs, ys
    return out
