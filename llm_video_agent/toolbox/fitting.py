from __future__ import annotations

from typing import Dict, Any
import numpy as np

try:
    from scipy.optimize import least_squares
except Exception:
    least_squares = None


def _fit_linear(t: np.ndarray, y: np.ndarray):
    A = np.vstack([t, np.ones_like(t)]).T
    m, b = np.linalg.lstsq(A, y, rcond=None)[0]
    return {'m': float(m), 'b': float(b)}, m*t + b


def _fit_quadratic(t: np.ndarray, y: np.ndarray):
    A = np.vstack([t**2, t, np.ones_like(t)]).T
    a, b, c = np.linalg.lstsq(A, y, rcond=None)[0]
    return {'a': float(a), 'b': float(b), 'c': float(c)}, a*t*t + b*t + c


def fit_projectile_xy(t, x, y, cfg: Dict[str, Any]):
    # x linear, y quadratic
    px, xfit = _fit_linear(t, x)
    if least_squares is not None and str(cfg.get('robust','none')).lower() == 'huber':
        delta = float(cfg.get('delta', 3.0))
        # robust y quadratic fit
        def model(p, tt): return p[0] + p[1]*tt + 0.5*p[2]*tt**2
        yy = y
        p0 = np.array([yy[0], (yy[1]-yy[0])/(t[1]-t[0]+1e-9), 120.0])
        def res(p):
            r = model(p, t) - yy
            a = np.abs(r)
            w = np.where(a <= delta, 1.0, delta/np.maximum(a,1e-9))
            return w*r
        p = least_squares(res, p0, max_nfev=20000).x
        py = {'a': float(p[2]/2.0*2), 'b': float(p[1]), 'c': float(p[0])}
        yfit = model(p, t)
    else:
        py, yfit = _fit_quadratic(t, y)
    return {'params': {'x': px, 'y': py}, 'x_fit': xfit, 'y_fit': yfit}


def fit_linear_xy(t, x, y, cfg: Dict[str, Any]):
    px, xfit = _fit_linear(t, x)
    py, yfit = _fit_linear(t, y)
    return {'params': {'x': px, 'y': py}, 'x_fit': xfit, 'y_fit': yfit}


def fit_free_fall(t, x, y, cfg: Dict[str, Any]):
    # x ~ const, y quadratic
    px, xfit = _fit_linear(t, x)
    py, yfit = _fit_quadratic(t, y)
    return {'params': {'x': px, 'y': py}, 'x_fit': xfit, 'y_fit': yfit}


def fit_uniform_accel_1d(t, x, y, cfg: Dict[str, Any]):
    # PCA along dominant axis, fit quadratic, project back
    pts = np.column_stack([x, y])
    mu = np.nanmean(pts, axis=0)
    pts_c = pts - mu
    cov = np.cov(pts_c.T)
    w, v = np.linalg.eig(cov)
    axis = v[:, np.argmax(w)]
    s = pts_c @ axis
    # quadratic fit on s(t)
    A = np.vstack([t**2, t, np.ones_like(t)]).T
    a, b, c = np.linalg.lstsq(A, s, rcond=None)[0]
    sfit = a*t*t + b*t + c
    xy_fit = np.outer(sfit, axis) + mu
    return {'params': {'axis': axis.tolist(), 'a': float(a), 'b': float(b), 'c': float(c)},
            'x_fit': xy_fit[:,0], 'y_fit': xy_fit[:,1]}


def fit_sho_xy(t, x, y, cfg: Dict[str, Any]):
    """Fit simple harmonic motion along the dominant axis.
    Model: s(t) = C + A*cos(omega*t + phi)
    Returns x_fit/y_fit with the other axis held at its mean.
    """
    vx = np.nanvar(x); vy = np.nanvar(y)
    s = x.copy() if vx >= vy else y.copy()
    other_mean = float(np.nanmean(y if vx >= vy else x))

    m = np.isfinite(t) & np.isfinite(s)
    tt = t[m]; ss = s[m]
    if len(tt) < 5:
        return fit_linear_xy(t, x, y, cfg)

    C0 = float(np.mean(ss))
    A0 = float((np.max(ss) - np.min(ss)) / 2.0) or 1.0
    dt = np.median(np.diff(tt)) if len(tt) > 1 else 1.0
    try:
        freqs = np.fft.rfftfreq(len(tt), d=dt)
        spec = np.abs(np.fft.rfft(ss - np.mean(ss)))
        if len(spec) > 1:
            k = int(np.argmax(spec[1:]) + 1)
            f0 = max(1e-3, float(freqs[k]))
        else:
            f0 = 0.5
    except Exception:
        f0 = 0.5
    omega0 = 2*np.pi*f0
    phi0 = 0.0

    if least_squares is None:
        C, A, omega, phi = C0, A0, omega0, phi0
    else:
        def model(p, tt):
            C, A, omega, phi = p
            return C + A * np.cos(omega*tt + phi)
        def resid(p):
            return model(p, tt) - ss
        p0 = np.array([C0, A0, omega0, phi0], dtype=float)
        try:
            res = least_squares(resid, p0, bounds=([-np.inf, -np.inf, 1e-4, -10*np.pi], [np.inf, np.inf, 50*np.pi, 10*np.pi]), max_nfev=20000)
            C, A, omega, phi = res.x
        except Exception:
            C, A, omega, phi = p0

    s_fit = C + A * np.cos(omega*t + phi)
    if vx >= vy:
        x_fit = s_fit
        y_fit = np.full_like(t, other_mean)
    else:
        x_fit = np.full_like(t, other_mean)
        y_fit = s_fit

    freq = float(omega/(2*np.pi)) if omega > 0 else None
    period = float((2*np.pi)/omega) if omega > 0 else None
    return {'params': {'C': float(C), 'A': float(A), 'omega': float(omega), 'phi': float(phi), 'axis': 'x' if vx>=vy else 'y', 'frequency': freq, 'period': period},
            'x_fit': x_fit, 'y_fit': y_fit}
