N, M, K = map(int, input().split())
MOD = 10**9+7

def makefactlist(n, mod=MOD):
    ret = [1]
    for i in range(n):
        ret += [ (ret[-1]*(i+1))%MOD ]
    return ret

factlist = makefactlist(N*M, mod=MOD)
comb = factlist[N*M-2]*pow(factlist[N*M-K], MOD-2, MOD)*pow(factlist[K-2], MOD-2, MOD)
comb = comb % MOD

ret = 0
for x in range(1,N):
    ret += ((N-x)*M*M*comb*x) % MOD
    ret = ret % MOD
for y in range(1,M):
    ret += ((M-y)*N*N*comb*y) % MOD
    ret = ret % MOD
print(ret)