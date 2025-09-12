from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from .prompt import PLANNER_PROMPT, PLANNER_PROMPT_COMPACT
from .schema import validate_and_normalize, PlanSpec


def _extract_json_object(text: str) -> Dict[str, Any] | None:
    # Strip code fences if present
    s = text.strip()
    if s.startswith("```"):
        s = s.strip('`')
        if s.lower().startswith("json"):
            s = s[4:]
    # Take the largest JSON object
    start = s.find('{'); end = s.rfind('}')
    if start != -1 and end != -1 and end > start:
        fragment = s[start:end+1]
        try:
            return json.loads(fragment)
        except Exception:
            return None
    return None


@dataclass
class PlannerClient:
    host: str = "http://127.0.0.1"
    port: int = 8000
    timeout_s: int = 120

    def plan(self, problem_desc: str, out_dir: Path) -> PlanSpec:
        try:
            import requests
        except Exception as e:
            raise RuntimeError("'requests' is required for planner backend. Please install it.") from e

        out_dir.mkdir(parents=True, exist_ok=True)
        url = f"{self.host}:{self.port}/completions"
        prompt = PLANNER_PROMPT.replace("<<PROMPT>>", problem_desc)
        payload = {
            "prompt": prompt,
            "params": {
                "max_new_tokens": 2048,
                "temperature": 0.1,
                "do_sample": False,
                "top_p": 0.9,
            },
            "repeat_prompt": 1,
        }
        resp = requests.post(url, json=payload, timeout=self.timeout_s)
        if resp.status_code != 200:
            raise RuntimeError(f"Planner LLM HTTP {resp.status_code}: {resp.text}")
        data = resp.json()
        texts = data.get("content", [])
        if not texts:
            raise RuntimeError("Planner LLM returned empty content")
        raw = texts[0]
        (out_dir / "planner_raw.txt").write_text(raw)
        obj = _extract_json_object(raw)
        # Retry with compact prompt if parsing failed
        if obj is None:
            prompt2 = PLANNER_PROMPT_COMPACT.replace("<<PROMPT>>", problem_desc)
            payload2 = {
                "prompt": prompt2,
                "params": {
                    "max_new_tokens": 2048,
                    "temperature": 0.0,
                    "do_sample": False,
                    "top_p": 1.0,
                },
                "repeat_prompt": 1,
            }
            resp2 = requests.post(url, json=payload2, timeout=self.timeout_s)
            if resp2.status_code == 200:
                data2 = resp2.json()
                texts2 = data2.get("content", [])
                if texts2:
                    raw2 = texts2[0]
                    (out_dir / "planner_raw_retry.txt").write_text(raw2)
                    obj = _extract_json_object(raw2)
        if obj is None:
            raise RuntimeError("Planner LLM did not return a valid JSON object")
        (out_dir / "plan.json").write_text(json.dumps(obj, indent=2))

        # Extract optional snippets from plan JSON if present
        snippets = obj.get("snippets", []) or []
        for sn in snippets:
            tgt = str(sn.get("target", ""))
            body = sn.get("body", "")
            if not tgt or not isinstance(body, str) or not body.strip():
                continue
            # Map target to filename
            if tgt.endswith("track_hsv_nn"):
                (out_dir / "snippet_detector.py").write_text(body)
            elif tgt.endswith("smooth_kalman"):
                (out_dir / "snippet_smooth.py").write_text(body)
            elif tgt.endswith("fit_projectile") or tgt.endswith("fit_motion"):
                (out_dir / "snippet_fit.py").write_text(body)

        # Validate and normalize plan
        spec = validate_and_normalize(obj)
        (out_dir / "plan.normalized.json").write_text(json.dumps({
            "plan_id": spec.plan_id,
            "tracker": spec.tracker,
            "denoiser": spec.denoiser,
            "fitter": spec.fitter,
            "masks": spec.masks,
            "overlays": spec.overlays,
            "snippets": spec.snippets,
        }, indent=2))
        return spec
