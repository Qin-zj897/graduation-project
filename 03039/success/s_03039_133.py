N, M, K = map(int, input().split())
mod = 10**9+7

# comb init
nmax = 2 * 10 ** 5 + 1  # change here
fac = [0] * (nmax + 1)
finv = [0] * (nmax + 1)
inv = [0] * (nmax + 1)
fac[0] = 1
fac[1] = 1
finv[0] = 1
finv[1] = 1
inv[1] = 1
for i in range(2, nmax):
    fac[i] = fac[i - 1] * i % mod
    inv[i] = mod - inv[mod % i] * (mod // i) % mod
    finv[i] = finv[i - 1] * inv[i] % mod


def comb(n, r):
    if n < r:
        return 0
    else:
        return (fac[n] * ((finv[r] * finv[n - r]) % mod)) % mod


def s(k):
    return ((k * (k+1) * (2 * k + 1))%mod * inv[6] + (k * (k+1) * inv[2])%mod)%mod


C = comb(N*M-2, K-2)
print(((C * ((N ** 2 * s(M-1))%mod + (M ** 2 * s(N-1))%mod)%mod)%mod) * inv[2]%mod)
