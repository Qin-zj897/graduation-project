mod = 10**9 + 7

def comb(n, k):
    x, y = 1, 1
    k = min(k, n-k)
    for i in range(k):
        x *= n - i
        x %= mod
        y *= i + 1
        y %= mod
    return x*pow(y, mod-2, mod) % mod

N, M, K = map(int, input().split())
print((M*M%mod*((N**3-N)//6%mod)%mod+N*N%mod*((M**3-M)//6%mod)%mod)%mod*comb(N*M-2,K-2)%mod)