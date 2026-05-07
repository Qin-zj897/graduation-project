# https://atcoder.jp/contests/abc127/submissions/5606400

from functools import reduce


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

ans_x = 0
for i in range(M):
    ans_x += i * (M - i)
ans_x *= N ** 2

ans_y = 0
for i in range(N):
    ans_y += i * (N - i)
ans_y *= M ** 2

ans = ans_x + ans_y

pat = cmb(N * M - 2, K - 2)
ans = (ans * pat) % MOD

print(ans)
