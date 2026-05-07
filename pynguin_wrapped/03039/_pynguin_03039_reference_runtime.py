from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ORIGINAL_REFERENCE = Path(r"C:\Users\22954\Desktop\毕业设计\datasets\03039\reference.py")
TIMEOUT_SECONDS = 8


def solve(n: int, m: int, k: int) -> int:
    if isinstance(n, bool) or isinstance(m, bool) or isinstance(k, bool):
        raise TypeError("Boolean values are not valid 03039 inputs")
    if not all(isinstance(value, int) for value in (n, m, k)):
        raise TypeError("03039 solve expects three integers")
    if n < 1 or m < 1 or k < 2 or k > n * m:
        raise ValueError("03039 inputs violate problem constraints")
    completed = subprocess.run(
        [sys.executable, str(ORIGINAL_REFERENCE)],
        input=f"{n} {m} {k}\n",
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
    try:
        return int(output)
    except ValueError as exc:
        raise ValueError(f"Non-integer output: {output!r}") from exc
