MAX = 2*10**5+100
MOD = 10**9+7
fact = [0]*MAX #fact[i]: i!
inv = [0]*MAX #inv[i]: iの逆元
finv = [0]*MAX #finv[i]: i!の逆元
fact[0] = 1
fact[1] = 1
finv[0] = 1
finv[1] = 1
inv[1] = 1
    
for i in range(2, MAX):
    fact[i] = fact[i-1]*i%MOD
    inv[i] = MOD-inv[MOD%i]*(MOD//i)%MOD
    finv[i] = finv[i-1]*inv[i]%MOD

def C(n, r):
    if n<r:
        return 0
    if n<0 or r<0:
        return 0
    return fact[n]*(finv[r]*finv[n-r]%MOD)%MOD

N, M, K = map(int, input().split())
ans = 0

for i in range(1, N):
    ans += i*(N-i)*M*M*C(N*M-2, K-2)
    ans %= MOD

for i in range(1, M):
    ans += i*(M-i)*N*N*C(N*M-2, K-2)
    ans %= MOD

print(ans)