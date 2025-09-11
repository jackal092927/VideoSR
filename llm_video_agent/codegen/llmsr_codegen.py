from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any

from .simple_codegen import SimpleCodeGenerator


SPEC_PROMPT = """
You are a code planning assistant. Given a short description of a simple synthetic video of a single, clearly-colored object undergoing basic physics motion, output ONLY a compact JSON configuration for a pre-defined analyzer template.

Constraints:
- Do NOT output code. Output JSON only.
- JSON keys: motion_class (one of: projectile, free_fall, constant_velocity, uniform_acceleration_1d),
             detector (one of: hsv, threshold, hough),
             hsv_lower (array of 3 ints [H,S,V] if detector=hsv),
             hsv_upper (array of 3 ints [H,S,V] if detector=hsv),
             min_area (int), morph_kernel (int).
- Assume the object color in prompt; pick robust HSV bounds with reasonable ranges.

Example JSON:
{"motion_class":"projectile","detector":"hsv","hsv_lower":[100,80,50],"hsv_upper":[130,255,255],"min_area":50,"morph_kernel":5}

Now infer for this prompt:
PROMPT: <<PROMPT>>
"""


@dataclass
class LLMSRCodeGenerator:
    host: str = "http://127.0.0.1"
    port: int = 8009

    def generate(self, video_path: Path, prompt: str, gen_dir: Path, options: Dict[str, Any] | None = None) -> Path:
        # Ask local LLMSR engine to produce JSON config
        try:
            import requests
        except Exception as e:
            raise RuntimeError("'requests' is required for llmsr backend. Install it in your env.") from e

        url = f"{self.host}:{self.port}/completions"
        payload = {
            "prompt": SPEC_PROMPT.replace("<<PROMPT>>", prompt),
            "params": {
                "max_new_tokens": 256,
                "temperature": 0.1,
                "do_sample": False,
                "top_p": 0.9,
            },
            "repeat_prompt": 1,
        }
        try:
            resp = requests.post(url, json=payload, timeout=120)
        except Exception as e:
            raise RuntimeError(f"Failed to reach LLMSR engine at {url}. Is it running?") from e
        if resp.status_code != 200:
            raise RuntimeError(f"LLMSR engine HTTP {resp.status_code}: {resp.text}")
        data = resp.json()
        texts = data.get("content", [])
        if not texts:
            raise RuntimeError("LLMSR engine returned empty content.")
        text = texts[0].strip()
        # Persist raw LLM output for inspection
        (gen_dir / "llm_raw.txt").write_text(text)
        # Try to parse JSON from the output
        cfg = None
        try:
            # Some models wrap JSON in code fences; strip them
            if text.startswith("```"):
                text = text.strip('`')
                # remove potential language tag
                if text.lower().startswith("json"):
                    text = text[4:]
            # Extract nearest JSON object
            start = text.find('{'); end = text.rfind('}')
            if start != -1 and end != -1 and end > start:
                text_json = text[start:end+1]
                cfg = json.loads(text_json)
        except Exception:
            cfg = None
        if not isinstance(cfg, dict):
            raise RuntimeError(f"LLMSR engine did not return valid JSON. Got: {text[:400]}")
        # Save parsed JSON for inspection
        (gen_dir / "llm_config.json").write_text(json.dumps(cfg, indent=2))

        # Fill defaults and route through SimpleCodeGenerator for actual script
        motion = cfg.get("motion_class") or SimpleCodeGenerator()._infer_motion_class(prompt)
        detector = (cfg.get("detector") or "hsv").lower()
        hsv_lower = cfg.get("hsv_lower")
        hsv_upper = cfg.get("hsv_upper")
        options = options or {}
        options.update({
            "detector": detector,
            "hsv_lower": hsv_lower,
            "hsv_upper": hsv_upper,
            "min_area": int(cfg.get("min_area", options.get("min_area", 50))),
            "morph_kernel": int(cfg.get("morph_kernel", options.get("morph_kernel", 5))),
        })
        gen = SimpleCodeGenerator()
        # Bypass motion inference: temporarily inject motion into prompt
        prompt_with_motion = f"{prompt}\n\n[motion_class={motion}]"
        return gen.generate(video_path=video_path, prompt=prompt_with_motion, gen_dir=gen_dir, options=options)
