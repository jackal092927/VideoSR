from pathlib import Path
from typing import Protocol, Dict, Any


class CodeGenerator(Protocol):
    def generate(self, video_path: Path, prompt: str, gen_dir: Path, options: Dict[str, Any] | None = None) -> Path:
        """Generate an analyzer script and return its path.
        Must be fully offline. The returned script must accept: --video, --outdir.
        """
        ...

