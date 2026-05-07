from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ORIGINAL_ROOT = Path(r"C:\Users\22954\Desktop\毕业设计\datasets\02882")
TIMEOUT_SECONDS = 8


def run_original(kind: str, module_name: str, input_text: str):
    program_path = ORIGINAL_ROOT / kind / f"{module_name}.py"
    completed = subprocess.run(
        [sys.executable, str(program_path)],
        input=input_text,
        text=True,
        capture_output=True,
        timeout=TIMEOUT_SECONDS,
        check=False,
    )
    if completed.returncode != 0:
        stderr = completed.stderr.strip()
        raise RuntimeError(stderr or f"Program exited with code {completed.returncode}")
    output = completed.stdout.strip()
    if not output:
        raise ValueError("Program produced empty output")
    return output
