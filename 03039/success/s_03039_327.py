mod = 10 ** 9 + 7
kkai = [1]
for i in range(1, 3 * 10 ** 5):
    kkai.append(kkai[-1] * i % mod)


def kai(x, p=mod):
    return kkai[x]


def comb(a, b, p=mod):
    if a < 0 or b < 0:
        return 0
    elif a < b:
        return 0
    c = 1
    c *= kai(a, p)
    c *= pow(kai(b, p), p - 2, p)
    c *= pow(kai(a - b, p), p - 2, p)
    return c % p


n, m, k = [int(i) for i in input().split()]

ans = 0
for i in range(1, n):
    ans += (n - i) * i * comb(n * m - 2, k - 2) * m ** 2
    ans %= mod
for i in range(1, m):
    ans += (m - i) * i * comb(n * m - 2, k - 2) * n ** 2
    ans %= mod
print(ans)
