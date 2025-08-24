import argparse
import yaml
from pathlib import Path
import pandas as pd

from extractor.extract import run_extraction
from discovery.sindy_fit import run_sindy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True, help="Path to input video")
    parser.add_argument("--config", default="configs/example.yaml")
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    Path("outputs").mkdir(parents=True, exist_ok=True)

    df, meta = run_extraction(args.video, cfg)

    csv_path = Path(cfg["output"]["csv_path"]).resolve()
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    print(f"[OK] Wrote {len(df)} rows → {csv_path}")

    if cfg.get("sindy", {}).get("enabled", True):
        eqn_txt = run_sindy(df, cfg.get("sindy", {}))
        out_txt = csv_path.with_suffix(".equations.txt")
        out_txt.write_text(eqn_txt)
        print("\n=== Discovered equations ===\n" + eqn_txt)


if __name__ == "__main__":
    main()