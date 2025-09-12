from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


ALLOWED_TRACKERS = {"hsv_nn", "contour", "grounded_sam2", "sa2va"}
ALLOWED_DENOISERS = {"savgol", "kalman", "none"}
ALLOWED_FITTERS = {"projectile", "sho", "linear", "free_fall", "uniform_accel_1d"}
ALLOWED_PICK_BY = {"AIC", "MSE"}


@dataclass
class PlanSpec:
    plan_id: str
    tracker: Dict[str, Any]
    denoiser: Dict[str, Any]
    fitter: Dict[str, Any]
    masks: Dict[str, Any] = field(default_factory=dict)
    overlays: Dict[str, Any] = field(default_factory=dict)
    snippets: List[Dict[str, Any]] = field(default_factory=list)


def _require(d: Dict[str, Any], key: str, typ):
    if key not in d:
        raise ValueError(f"Missing required key: {key}")
    v = d[key]
    if typ is list:
        if not isinstance(v, list):
            raise ValueError(f"{key} must be a list")
    elif typ is dict:
        if not isinstance(v, dict):
            raise ValueError(f"{key} must be a dict")
    elif not isinstance(v, typ):
        raise ValueError(f"{key} must be {typ}")
    return v


def validate_and_normalize(plan: Dict[str, Any]) -> PlanSpec:
    # Required top-level fields
    plan_id = str(_require(plan, "plan_id", str))
    tracker = dict(_require(plan, "tracker", dict))
    denoiser = dict(_require(plan, "denoiser", dict))
    fitter = dict(_require(plan, "fitter", dict))
    masks = dict(plan.get("masks", {}))
    overlays = dict(plan.get("overlays", {}))
    snippets = list(plan.get("snippets", []))

    # Tracker
    name = str(tracker.get("name", "hsv_nn")).lower()
    if name not in ALLOWED_TRACKERS:
        name = "hsv_nn"
    params = tracker.get("params", {}) or {}
    tracker = {"name": name, "params": params}

    # Denoiser
    dname = str(denoiser.get("name", "savgol")).lower()
    if dname not in ALLOWED_DENOISERS:
        dname = "savgol"
    dparams = denoiser.get("params", {}) or {}
    denoiser = {"name": dname, "params": dparams}

    # Fitter
    cand = fitter.get("candidates", []) or []
    cand = [c for c in cand if str(c) in ALLOWED_FITTERS]
    if not cand:
        cand = ["projectile", "linear"]
    pick_by = str(fitter.get("pick_by", "AIC"))
    if pick_by not in ALLOWED_PICK_BY:
        pick_by = "AIC"
    robust = str(fitter.get("robust", "none")).lower()
    delta = float(fitter.get("delta", 3.0))
    interior_mask = fitter.get("interior_mask", {}) or {}
    fitter = {
        "candidates": cand,
        "pick_by": pick_by,
        "robust": robust,
        "delta": delta,
        "interior_mask": {
            "radius": int(interior_mask.get("radius", 12)),
            "pad": int(interior_mask.get("pad", 5)),
        },
    }

    # Masks / overlays defaults
    border = (masks.get("border") or {})
    masks = {
        "border": {
            "radius": int(border.get("radius", 12)),
            "pad": int(border.get("pad", 5)),
        },
        "occlusion": (masks.get("occlusion") or {}),
    }
    overlays = {
        "fps": int(overlays.get("fps", 8)),
        "show_smoothed": bool(overlays.get("show_smoothed", True)),
        "show_fit": bool(overlays.get("show_fit", True)),
    }

    # Snippets: do only minimal shape validation here
    norm_snips: List[Dict[str, Any]] = []
    for sn in snippets:
        if not isinstance(sn, dict):
            continue
        tgt = str(sn.get("target", ""))
        lang = str(sn.get("language", "python"))
        body = sn.get("body", "")
        if tgt and isinstance(body, str) and lang.lower() == "python":
            norm_snips.append({"target": tgt, "language": "python", "body": body})

    return PlanSpec(
        plan_id=plan_id,
        tracker=tracker,
        denoiser=denoiser,
        fitter=fitter,
        masks=masks,
        overlays=overlays,
        snippets=norm_snips,
    )

