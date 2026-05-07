from __future__ import annotations

from _pynguin_03039_runtime import run_original

MODULE_NAME = "f_03039_189"
KIND = "failure"


def solve(n: int, m: int, k: int) -> int:
    if isinstance(n, bool) or isinstance(m, bool) or isinstance(k, bool):
        raise TypeError("Boolean values are not valid 03039 inputs")
    if not all(isinstance(value, int) for value in (n, m, k)):
        raise TypeError("03039 solve expects three integers")
    if n < 1 or m < 1 or k < 2 or k > n * m:
        raise ValueError("03039 inputs violate problem constraints")
    return run_original(KIND, MODULE_NAME, n, m, k)


def main() -> None:
    n, m, k = map(int, input().split())
    print(solve(n, m, k))


if __name__ == "__main__":
    main()
