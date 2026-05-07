from __future__ import annotations

from _pynguin_03039_reference_runtime import solve


def main() -> None:
    n, m, k = map(int, input().split())
    print(solve(n, m, k))


if __name__ == "__main__":
    main()
