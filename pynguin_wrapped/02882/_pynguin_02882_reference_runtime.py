from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ORIGINAL_REFERENCE = Path(r"C:\Users\22954\Desktop\毕业设计\datasets\02882\reference.py")
TIMEOUT_SECONDS = 8


def run_reference(input_text: str):
    completed = subprocess.run(
        [sys.executable, str(ORIGINAL_REFERENCE)],
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
