def comb(n, k, mod=10**9+7):
    num = den = 1
    for i in range(k):
        num = num * (n-i) % mod
        den = den * (i+1) % mod
    return num * pow(den, mod-2, mod) % mod

N, M, K = map(int, input().split())
mod = 10**9 + 7
ans = 0
for i in range(N):
    for j in range(M):
        ans += (i+1)*j*(j+1)//2 + i*(M-j-1)*(M-j)//2 + M*i*(i+1)//2
        ans %= mod
ans = ans * comb(N*M-2, K-2) % mod
print(ans)

