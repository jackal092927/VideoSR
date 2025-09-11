import argparse
from pathlib import Path
from .controller import AgentController


def main():
    ap = argparse.ArgumentParser(description="LLM Video Agent (template-based MVP)")
    ap.add_argument("--video", required=True, help="Input video path")
    ap.add_argument("--prompt", required=True, help="Motion description prompt")
    ap.add_argument("--outdir", default="./agent_out", help="Output directory")
    # Backend selection
    ap.add_argument("--backend", choices=["simple", "llmsr"], default="simple", help="Codegen backend")
    ap.add_argument("--llm-host", default="http://127.0.0.1", help="LLM server host (for llmsr backend)")
    ap.add_argument("--llm-port", type=int, default=8009, help="LLM server port (for llmsr backend)")
    # Optional detector overrides
    ap.add_argument("--detector", choices=["hsv", "threshold", "hough"], default=None)
    ap.add_argument("--hsv-lower", nargs=3, type=int, default=None, help="HSV lower bounds e.g. 100 80 50")
    ap.add_argument("--hsv-upper", nargs=3, type=int, default=None, help="HSV upper bounds e.g. 130 255 255")
    args = ap.parse_args()

    controller = AgentController()
    report = controller.run(
            video_path=Path(args.video),
            prompt=args.prompt,
            outdir=Path(args.outdir),
            options={
            "backend": args.backend,
            "llm_host": args.llm_host,
            "llm_port": args.llm_port,
            "detector": args.detector,
            "hsv_lower": args.hsv_lower,
            "hsv_upper": args.hsv_upper,
        },
    )
    # Print a brief summary
    print("[AGENT] Done. Summary:")
    for k, v in report.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
