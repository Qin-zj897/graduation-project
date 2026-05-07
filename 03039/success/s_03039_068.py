"""
https://atcoder.jp/contests/abc127/submissions/5606400
"""


def inv(x):
    return pow(x, MOD - 2, MOD)


def cmb(n, r):
    r = min(r, n - r)
    if r == 0:
        return 1
    if r == 1:
        return n

    f = 1
    for k in range(n, n - r, -1):
        f = (f * k) % MOD
        # n!/(n-r)!

    d = 1
    for k in range(2, r + 1):
        d = (d * k) % MOD
        # r!

    return (f * inv(d)) % MOD


MOD = 10 ** 9 + 7

N, M, K = map(int, input().split())

ans = 0

for i in range(M):
    ans += i * (M - i) * N ** 2

for i in range(N):
    ans += i * (N - i) * M ** 2

pat = cmb(N * M - 2, K - 2)
ans = (ans * pat) % MOD

print(ans)
