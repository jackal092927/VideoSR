import json
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any

from .codegen.simple_codegen import SimpleCodeGenerator
try:
    from .codegen.llmsr_codegen import LLMSRCodeGenerator
except Exception:
    LLMSRCodeGenerator = None
try:
    from .codegen.hybrid_codegen import HybridCodeGenerator
except Exception:
    HybridCodeGenerator = None
from .runner import run_generated_analyzer
from .planner.client import PlannerClient
from .evaluator import evaluate_outputs


@dataclass
class AgentController:
    def run(self, video_path: Path, prompt: str, outdir: Path, options: Dict[str, Any] | None = None) -> Dict[str, Any]:
        options = options or {}
        outdir.mkdir(parents=True, exist_ok=True)

        # Workspace for generated code
        gen_dir = outdir / "_generated"
        if gen_dir.exists():
            shutil.rmtree(gen_dir)
        gen_dir.mkdir(parents=True, exist_ok=True)

        # 1) Select backend and optionally plan
        backend = (options.get("backend") or "simple").lower()
        plan_meta = {}
        if backend == "llmsr_hybrid":
            # Run planner first; save artifacts. For now, still use llmsr/simple codegen.
            planner = PlannerClient(host=options.get("llm_host", "http://127.0.0.1"),
                                    port=int(options.get("llm_port", 8000)))
            try:
                spec = planner.plan(problem_desc=prompt, out_dir=gen_dir)
                plan_meta = {
                    "backend": backend,
                    "plan_id": spec.plan_id,
                    "tracker": spec.tracker.get("name"),
                    "denoiser": spec.denoiser.get("name"),
                    "fitter_candidates": spec.fitter.get("candidates"),
                }
            except Exception as e:
                plan_meta = {"backend": backend, "plan_error": str(e)}

        # 2) Codegen (template-based or llmsr)
        if backend == "llmsr" and LLMSRCodeGenerator is not None:
            generator = LLMSRCodeGenerator(
                host=options.get("llm_host", "http://127.0.0.1"),
                port=int(options.get("llm_port", 8009)),
            )
        elif backend == "llmsr_hybrid" and HybridCodeGenerator is not None:
            generator = HybridCodeGenerator()
        else:
            generator = SimpleCodeGenerator()
        analyzer_path = generator.generate(video_path=video_path, prompt=prompt, gen_dir=gen_dir, options=options)

        # 3) Run generated analyzer
        # For hybrid, analyzer expects a --plan path; runner passes only video/outdir.
        # We embed plan path read inside analyzer, so normal runner suffices.
        run_generated_analyzer(analyzer_path=analyzer_path, video_path=video_path, outdir=outdir)

        # 4) Evaluate results
        report = evaluate_outputs(outdir)
        # Attach backend/plan info if available
        if backend:
            report["backend"] = backend
        for k, v in (plan_meta or {}).items():
            report[k] = v
        # Attach tracker diagnostics if present
        tip = outdir / "tracker_info.json"
        if tip.exists():
            try:
                report["tracker_info"] = json.loads(tip.read_text())
            except Exception:
                pass
        # Attach effective plan info after overrides
        try:
            eff = json.loads((gen_dir / 'plan.normalized.json').read_text())
            report['effective_tracker'] = eff.get('tracker',{}).get('name')
            report['effective_denoiser'] = eff.get('denoiser',{}).get('name')
            overlays = eff.get('overlays',{})
            if isinstance(overlays, dict):
                report['effective_show_fit'] = overlays.get('show_fit')
        except Exception:
            pass
        (outdir / "agent_report.json").write_text(json.dumps(report, indent=2))
        return report
