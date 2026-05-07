from math import factorial

N, M, K = map(int, input().split())
mod = 10 ** 9 + 7

def Comb(N, K):
    a = factorial(N) % mod
    b = factorial(K) % mod
    c = factorial(N-K) % mod
    return (a * pow(b, mod-2, mod) * pow(c, mod-2, mod)) % mod


c = Comb(N*M-2, K-2)

ans = 0

for d in range(M):
    ans += d * (M - d) * (N ** 2)
    ans %= mod

for d in range(N):
    ans += d * (N - d) * (M ** 2)
    ans %= mod

ans *= c
ans %= mod

print(ans)