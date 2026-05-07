#!/usr/bin/env python3
import sys
INF = float("inf")

MOD = 1000000007  # type: int


def cmb(n, r):
    r = min(r, n-r)
    if r == 0:
        return 1
    if r == 1:
        return n

    numer = [n - r + k + 1 for k in range(r)]
    denom = [k + 1 for k in range(r)]

    for p in range(2, r+1):
        pivot = denom[p - 1]
        if pivot > 1:
            offset = (n - r) % p
            for k in range(p-1, r, p):
                numer[k - offset] /= pivot
                denom[k] /= pivot

    result = 1
    for k in range(r):
        if numer[k] > 1:
            result *= int(numer[k])
            result %= MOD

    return result


def solve(N: int, M: int, K: int):

    sx = (M*(M+1)*(M-1)//6) % MOD
    sx = (sx * N*N) % MOD
    sy = (N*(N+1)*(N-1)//6) % MOD
    sy = (sy * M*M) % MOD
    s = (sx + sy) % MOD
    s = (s*cmb(N*M-2, K-2)) % MOD
    print(s)

    return


def main():

    def iterate_tokens():
        for line in sys.stdin:
            for word in line.split():
                yield word
    tokens = iterate_tokens()
    N = int(next(tokens))  # type: int
    M = int(next(tokens))  # type: int
    K = int(next(tokens))  # type: int
    solve(N, M, K)


if __name__ == '__main__':
    main()
