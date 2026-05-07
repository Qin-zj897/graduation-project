# nCk ( O(N) )
N_MAX = 10**6
MOD = 10**9 + 7
fac, finv, inv = [0]*N_MAX ,[0]*N_MAX, [0]*N_MAX
def com_init():
    fac[0], fac[1] = 1, 1
    finv[0], finv[1] = 1, 1
    inv[1] = 1
    for i in range(2, N_MAX):
        fac[i] = fac[i - 1] * i % MOD
        inv[i] = MOD - inv[MOD%i] * (MOD // i) % MOD
        finv[i] = finv[i - 1] * inv[i] % MOD

def com(n, k):
    if n < k: return 0
    if n < 0 or k < 0: return 0
    return fac[n] * (finv[k] * finv[n - k] % MOD) % MOD


n,m,k = map(int, input().split())
com_init()

ans = 0
for dist in range(1,m):
    comb = (m-dist)*n*n
    other_comb = com(n*m-2,k-2)
    ans += comb*other_comb*dist
    ans %= MOD

for dist in range(1,n):
    comb = (n-dist)*m*m
    other_comb = com(n*m-2,k-2)
    ans += comb*other_comb*dist
    ans %= MOD

print(ans)