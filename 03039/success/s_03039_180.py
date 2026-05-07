from collections import defaultdict

mod = 10 ** 9 + 7

memo = defaultdict(int)


def cmb(n, r):
    if (n, r) in memo:
        return memo[(n, r)]

    if n < r or r < 0:
        return 0

    if r > n - r:
        return cmb(n, n - r)

    res = f[n] * pow(f[n - r], mod - 2, mod) * pow(f[r], mod - 2, mod) % mod
    memo[(n, r)] = res
    return res


n, m, k = map(int, input().split())

f = [1]
for i in range(1, 2 * 10 ** 5 + 1):
    f.append(f[-1] * i % mod)

ans = 0
for d in range(1, m):
    ans += d * (m - d) * n ** 2
    ans %= mod

for d in range(1, n):
    ans += d * (n - d) * m ** 2
    ans %= mod

ans *= cmb(n * m - 2, k - 2)
ans %= mod

print(ans)
