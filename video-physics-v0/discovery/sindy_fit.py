import numpy as np
import pandas as pd
import pysindy as ps


def run_sindy(df: pd.DataFrame, cfg: dict) -> str:
    # Use only rows where x,y are finite
    data = df[["t", "x", "y"]].dropna().to_numpy()
    t = data[:, 0]
    X = data[:, 1:3]

    # Uniformize dt if needed for SINDy finite differences
    # (SINDy supports uneven t through smoothed FD if desired.)

    library = ps.PolynomialLibrary(
        degree=int(cfg.get("poly_degree", 3)), 
        include_interaction=True, 
        include_bias=True
    )
    optimizer = ps.STLSQ(
        threshold=float(cfg.get("thresh", 0.01)), 
        alpha=0.0
    )
    feature_names = ["x", "y"]

    model = ps.SINDy(
        feature_library=library,
        optimizer=optimizer,
        discrete_time=bool(cfg.get("discrete_time", False)),
    )

    # Let SINDy estimate derivatives from X and t
    model.fit(X, t=t)

    # Pretty-print equations
    eqn_txt = model.equations()
    return "\n".join(eqn_txt)