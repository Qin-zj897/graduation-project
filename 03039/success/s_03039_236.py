# https://atcoder.jp/contests/abc127/submissions/5606400

from functools import reduce
from operator import add


def mod_mul(a, b):
    return (a * b) % MOD


def cmb(n, r):
    r = min(r, n - r)
    if r == 0:
        return 1
    if r == 1:
        return n
    f = reduce(mod_mul, range(n - r + 1, n + 1))  # n!/(n-r)!
    d = reduce(mod_mul, range(2, r + 1))  # r!
    return (f * pow(d, MOD - 2, MOD)) % MOD


MOD = 10 ** 9 + 7

N, M, K = map(int, input().split())

ans_x = reduce(add, (i * (M - i) for i in range(M))) * N ** 2
ans_y = reduce(add, (i * (N - i) for i in range(N))) * M ** 2
ans = (ans_x + ans_y) % MOD
pat = cmb(N * M - 2, K - 2)
ans = (ans * pat) % MOD

print(ans)
