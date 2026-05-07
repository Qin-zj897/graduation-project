N, M, K = map(int, input().split())
MOD = 10 ** 9 + 7

def comb(n, k):
    ret = 1
    for i in range(1, k + 1):
        ret = ret * (n - i + 1) % MOD        
        ret = ret * pow(i, MOD - 2, MOD) % MOD
        
    return ret

ans = 0
for d in range(N):
    ans += d * (N - d) * M ** 2
    ans %= MOD
    
for d in range(M):
    ans += d * (M - d) * N ** 2
    ans %= MOD
    
print(ans * comb(N * M - 2, K - 2) % MOD)

