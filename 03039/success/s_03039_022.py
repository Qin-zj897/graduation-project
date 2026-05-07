N,M,K = map(int,input().split())
MOD = 10**9+7

a = 0
for i in range(N):
    j = N-1-i
    a += i*(i+1)//2 + j*(j+1)//2

b = 0
for i in range(M):
    j = M-1-i
    b += i*(i+1)//2 + j*(j+1)//2

MAX = N*M+5
fac = [1,1] + [0]*MAX
finv = [1,1] + [0]*MAX
inv = [0,1] + [0]*MAX
for i in range(2,MAX+2):
    fac[i] = fac[i-1] * i % MOD
    inv[i] = -inv[MOD%i] * (MOD // i) % MOD
    finv[i] = finv[i-1] * inv[i] % MOD

def ncr(n,r):
    if n < r: return 0
    if n < 0 or r < 0: return 0
    return fac[n] * (finv[r] * finv[n-r] % MOD) % MOD

ans = (a * inv[N]**2 + b * inv[M]**2) * inv[N*M-1] * (N*M)
ans *= K*(K-1)//2
ans *= ncr(N*M, K)
print(ans % MOD)