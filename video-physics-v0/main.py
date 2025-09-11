import argparse
import yaml
from pathlib import Path
import pandas as pd
from datetime import datetime
import cv2

from extractor.extract import run_extraction
from discovery.sindy_fit import run_sindy
from extractor.overlay import write_overlay


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True, help="Path to input video")
    parser.add_argument("--config", default="configs/example.yaml")
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    Path("outputs").mkdir(parents=True, exist_ok=True)

    # Run extraction without overlay first
    df, meta = run_extraction(args.video, cfg)

    # Generate unique filename with timestamp to avoid overwriting
    base_path = Path(cfg["output"]["csv_path"])
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Split filename and extension
    stem = base_path.stem
    suffix = base_path.suffix
    
    # Create unique filename with timestamp
    unique_filename = f"{stem}_{timestamp}{suffix}"
    csv_path = base_path.parent / unique_filename
    
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    print(f"[OK] Wrote {len(df)} rows → {csv_path}")

    # Discover physics equations
    equations = []
    if cfg.get("sindy", {}).get("enabled", True):
        eqn_txt = run_sindy(df, cfg.get("sindy", {}))
        out_txt = csv_path.with_suffix(".equations.txt")
        out_txt.write_text(eqn_txt)
        print("\n=== Discovered equations ===\n" + eqn_txt)
        
        # Parse equations for overlay
        equations = eqn_txt.strip().split('\n')
    
    # Generate overlay video with equations
    if equations:
        print(f"\nGenerating overlay video with {len(equations)} discovered equations...")
        # Re-open video for overlay generation
        cap = cv2.VideoCapture(args.video)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {args.video}")
        
        # Generate unique overlay filename
        overlay_base = Path(cfg["output"]["overlay_path"])
        overlay_unique = overlay_base.parent / f"{overlay_base.stem}_{timestamp}{overlay_base.suffix}"
        overlay_unique.parent.mkdir(parents=True, exist_ok=True)
        
        # Generate overlay with equations
        write_overlay(cap, df, str(overlay_unique), meta["width"], meta["height"], meta["fps"], equations)
        cap.release()
        
        print(f"[OK] Generated overlay video → {overlay_unique}")


if __name__ == "__main__":
    main()