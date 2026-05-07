import operator as op
from functools import reduce


def ncr(n, r):
    r = min(r, n - r)
    numer = reduce(op.mul, range(n, n - r, -1), 1) % (10**9+7)
    denom = reduce(op.mul, range(1, r + 1), 1)
    return numer // denom


def solve(string):
    n, m, k = map(int, string.split())
    base = n * m * ((n + 1) * (m + 1) * (n + m - 2) - n**2 - m**2 + 2) // 6 % (10**9 + 7)
    return str(base * ncr(m * n - 2, k - 2) % (10**9 + 7))


if __name__ == '__main__':
    print(solve(input()))
