from __future__ import annotations

from _pynguin_03025_reference_runtime import run_reference


def solve(n: int, a: int, b: int, c: int) -> int:
    if any(isinstance(value, bool) for value in (n, a, b, c)):
        raise TypeError("Boolean values are not valid 03025 inputs")
    if not all(isinstance(value, int) for value in (n, a, b, c)):
        raise TypeError("03025 solve expects four integers")
    if n < 1 or n > 200:
        raise ValueError("03025 n violates generation constraints")
    if a < 0 or b < 0 or c < 0:
        raise ValueError("03025 probabilities must be non-negative")
    if a + b + c != 100:
        raise ValueError("03025 probabilities must sum to 100")
    if a + b == 0:
        raise ValueError("03025 requires a + b > 0")
    output = run_reference(f"{n} {a} {b} {c}\n")
    try:
        return int(output)
    except ValueError as exc:
        raise ValueError(f"Unexpected output: {output!r}") from exc


def solve_normalized(n: int, a: int, b: int) -> int:
    if any(isinstance(value, bool) for value in (n, a, b)):
        raise TypeError("Boolean values are not valid 03025 inputs")
    if not all(isinstance(value, int) for value in (n, a, b)):
        raise TypeError("03025 solve_normalized expects three integers")
    n = abs(n) % 200 + 1
    a = abs(a) % 100
    b = abs(b) % 100
    if a + b == 0:
        a = 1
    if a + b > 100:
        total = a + b
        a = max(1, a * 100 // total)
        b = 100 - a
        if b == 0 and total > 1:
            b = 1
            a = 99
    c = 100 - a - b
    return solve(n, a, b, c)


def main() -> None:
    n, a, b, c = map(int, input().split())
    print(solve(n, a, b, c))


if __name__ == "__main__":
    main()
