import subprocess
import sys
from pathlib import Path


def run_generated_analyzer(analyzer_path: Path, video_path: Path, outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, str(analyzer_path), "--video", str(video_path), "--outdir", str(outdir)]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"Generated analyzer failed with code {proc.returncode}:\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}")

