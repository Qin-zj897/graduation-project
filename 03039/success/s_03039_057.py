n, m, k = [int(i) for i in input().split()]
p = 10 ** 9 + 7

S = 0

for d in range(n):
    S += m ** 2 * d * (n - d)
    S %= p

for d in range(m):
    S += n ** 2 * d * (m - d)
    S %= p

def fact(n, p=10**9 + 7):
    f = [1]
    for i in range(1, n+1):
        f.append(f[-1]*i%p)
    return f

def invfact(n, f, p=10**9 + 7):
    inv = [pow(f[n], p-2, p)]
    for i in range(n, 0, -1):
        inv.append(inv[-1]*i%p)
    return inv[::-1]

f = fact(n * m - 2)
invf = invfact(n * m - 2, f)

def comb(a, b):
    if a < b:
        return 0
    if a < 0 or b < 0:
        return 0
    return f[a] * invf[b] * invf[a-b] % p

S *= comb(n * m - 2, k - 2)

print(S % p)