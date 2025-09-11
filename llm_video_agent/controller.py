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
from .runner import run_generated_analyzer
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

        # 1) Codegen (template-based MVP)
        backend = (options.get("backend") or "simple").lower()
        if backend == "llmsr" and LLMSRCodeGenerator is not None:
            generator = LLMSRCodeGenerator(
                host=options.get("llm_host", "http://127.0.0.1"),
                port=int(options.get("llm_port", 8009)),
            )
        else:
            generator = SimpleCodeGenerator()
        analyzer_path = generator.generate(video_path=video_path, prompt=prompt, gen_dir=gen_dir, options=options)

        # 2) Run generated analyzer
        run_generated_analyzer(analyzer_path=analyzer_path, video_path=video_path, outdir=outdir)

        # 3) Evaluate results
        report = evaluate_outputs(outdir)
        (outdir / "agent_report.json").write_text(json.dumps(report, indent=2))
        return report
