import argparse
from pathlib import Path
from .controller import AgentController


def main():
    ap = argparse.ArgumentParser(description="LLM Video Agent (template-based MVP)")
    ap.add_argument("--video", required=True, help="Input video path")
    ap.add_argument("--prompt", required=True, help="Motion description prompt")
    ap.add_argument("--outdir", default="./agent_out", help="Output directory")
    # Backend selection
    ap.add_argument("--backend", choices=["simple", "llmsr", "llmsr_hybrid"], default="simple", help="Codegen backend")
    ap.add_argument("--llm-host", default="http://127.0.0.1", help="LLM server host (for llmsr backend)")
    ap.add_argument("--llm-port", type=int, default=8009, help="LLM server port (for llmsr backend)")
    # Optional detector overrides
    ap.add_argument("--detector", choices=["hsv", "threshold", "hough"], default=None)
    ap.add_argument("--hsv-lower", nargs=3, type=int, default=None, help="HSV lower bounds e.g. 100 80 50")
    ap.add_argument("--hsv-upper", nargs=3, type=int, default=None, help="HSV upper bounds e.g. 130 255 255")
    # Hybrid convenience toggles
    ap.add_argument("--overlay-no-fit", action="store_true", help="Do not draw fitted trajectory in overlay")
    ap.add_argument("--auto-hsv", action="store_true", help="Use auto HSV calibration (ignore explicit bounds)")
    ap.add_argument("--denoiser", choices=["none", "savgol", "kalman"], default="savgol", help="Denoiser override for hybrid backend")
    # Hybrid tracking refinement and gating
    ap.add_argument("--calibrate-refine", action="store_true", help="Enable small deterministic HSV refinement search (hybrid)")
    ap.add_argument("--refine-hue-deg", type=int, default=5, help="Hue center shift for refinement (deg)")
    ap.add_argument("--refine-v-delta", type=int, default=20, help="Increase V lower bound by this amount in refinement")
    ap.add_argument("--refine-frames", type=int, default=10, help="Frames to evaluate during refinement search")
    ap.add_argument("--confirm-ball", action="store_true", help="Prefer circular blobs via Hough/circularity gating (hybrid)")
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
            "overlay_no_fit": args.overlay_no_fit,
            "auto_hsv": args.auto_hsv,
            "denoiser": args.denoiser,
            # refinement + shape-aware options (hybrid)
            "calibrate_refine": args.calibrate_refine,
            "refine_hue_deg": args.refine_hue_deg,
            "refine_v_delta": args.refine_v_delta,
            "refine_frames": args.refine_frames,
            "confirm_ball": args.confirm_ball,
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
