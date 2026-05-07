from math import factorial

N, M, K = map(int, input().split())
mod = 10 ** 9 + 7

def f(N):
    a = 1
    for i in range(1, N+1):
        a *= i
        a %= mod
    return a % mod

def Comb(N, K):
    a = f(N)
    b = f(K)
    c = f(N-K)
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