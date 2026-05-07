from __future__ import annotations

from _pynguin_02882_runtime import run_original

MODULE_NAME = "s_02882_112"
KIND = "success"


def solve(a: int, b: int, x: int) -> float:
    if isinstance(a, bool) or isinstance(b, bool) or isinstance(x, bool):
        raise TypeError("Boolean values are not valid 02882 inputs")
    if not all(isinstance(value, int) for value in (a, b, x)):
        raise TypeError("02882 solve expects three integers")
    if a < 1 or b < 1 or x < 1 or x > a * a * b:
        raise ValueError("02882 inputs violate problem constraints")
    output = run_original(KIND, MODULE_NAME, f"{a} {b} {x}\n")
    try:
        return float(output)
    except ValueError as exc:
        raise ValueError(f"Unexpected output: {output!r}") from exc


def main() -> None:
    a, b, x = map(int, input().split())
    print(solve(a, b, x))


if __name__ == "__main__":
    main()
