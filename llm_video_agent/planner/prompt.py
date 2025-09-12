PLANNER_PROMPT = """
You are a planning assistant for a video physics analyzer. Produce a SINGLE JSON object only, no prose.
The JSON must conform to this schema (keys and allowed values):
{
  "plan_id": string,
  "tracker": {"name": "hsv_nn|contour|grounded_sam2|sa2va", "params": object},
  "denoiser": {"name": "savgol|kalman|none", "params": object},
  "fitter": {
    "candidates": ["projectile","sho","linear","free_fall","uniform_accel_1d"],
    "pick_by": "AIC|MSE",
    "robust": "none|huber",
    "delta": number,
    "interior_mask": {"radius": integer, "pad": integer}
  },
  "masks": {"border": {"radius": integer, "pad": integer}, "occlusion": object},
  "overlays": {"fps": integer, "show_smoothed": boolean, "show_fit": boolean},
  "snippets": [
    {"target": "toolbox.tracking.track_hsv_nn|toolbox.denoise.smooth_kalman|toolbox.fitting.fit_projectile",
     "language": "python", "body": "def ...: ..."}
  ]
}

Guidance:
- Use information in the PROBLEM DESCRIPTION to set tracker params (e.g., color HSV ranges), pick candidates, and noise handling.
- Keep snippets optional; include at most TWO short python functions (<=40 lines each) if clearly beneficial.
- If environment tools like grounded_sam2/sa2va are not available, you may still name them but prefer hsv_nn/contour for now.

Now write ONLY the JSON (no code fences) for this problem:
PROBLEM DESCRIPTION:
<<PROMPT>>
"""

# A compact variant to reduce verbosity and avoid truncation.
PLANNER_PROMPT_COMPACT = """
Output ONE JSON object on a single line (no code fences, no extra text). Keys:
{"plan_id":string,
 "tracker":{"name":"hsv_nn|contour|grounded_sam2|sa2va","params":object},
 "denoiser":{"name":"savgol|kalman|none","params":object},
 "fitter":{"candidates":["projectile","sho","linear","free_fall","uniform_accel_1d"],"pick_by":"AIC|MSE","robust":"none|huber","delta":number,"interior_mask":{"radius":int,"pad":int}},
 "masks":{"border":{"radius":int,"pad":int}},
 "overlays":{"fps":int,"show_smoothed":bool,"show_fit":bool},
 "snippets":[]}
Use short values. End with '}'.
PROBLEM: <<PROMPT>>
"""
