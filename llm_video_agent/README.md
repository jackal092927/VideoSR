LLM Video Agent (MVP)

Goal: Given a simple synthetic video (single, clearly colored object) and a natural-language motion description (e.g., "blue ball doing projectile motion"), dynamically generate and run analysis code to:
- detect/track the object (HSV/threshold/Hough),
- extract trajectory, smooth and compute derivatives,
- fit simple physics models (projectile, free fall, constant velocity, uniform acceleration 1D),
- and output CSV, annotated video, plots, and fit_params.json — similar to videoXer/video-physics-v0.

This MVP is offline and pluggable: it uses a template-based code generator by default and can later be wired to LLM-SR’s local engine (e.g., Mixtral 8x7B).

Quick start
- Synthetic video: `python -m llm_video_agent.synth.generate --out demo.mp4 --motion projectile --color blue`
- Run agent: `python -m llm_video_agent.main --video demo.mp4 --prompt "blue ball doing projectile motion" --outdir ./agent_out`

Outputs
- outdir/measurements.csv
- outdir/annotated.mp4
- outdir/plots/*.png
- outdir/fit_params.json

Structure
- llm_video_agent/main.py            # CLI entrypoint
- llm_video_agent/controller.py      # Orchestrates prompt -> codegen -> run -> evaluate
- llm_video_agent/runner.py          # Executes generated analyzer script
- llm_video_agent/evaluator.py       # Basic success metrics (R^2, detection rate)
- llm_video_agent/codegen/
  - interface.py                     # CodeGenerator protocol
  - simple_codegen.py                # Offline template-based generator
- llm_video_agent/templates/analyzer_template.py  # Analyzer script template
- llm_video_agent/synth/generate.py  # Simple synthetic video generator

Notes
- Offline by default. No network access is required.
- Pluggable LLM backend: replace `simple_codegen.SimpleCodeGenerator` with an adapter that calls a local LLM server (e.g., LLM-SR-fork engine) and fills the same template constraints.

