MOD = 10**9+7
N,M,K = map(int,input().split())

U = N*M
fact = [1]*(U+1)
for i in range(1,U+1):
    fact[i] = (fact[i-1]*i)%MOD

def comb(n,k):
    x = fact[n]
    x *= pow(fact[n-k],MOD-2,MOD)
    x %= MOD
    x *= pow(fact[k],MOD-2,MOD)
    return x

res = 0

for i in range(1,N):
    res += i*M*M*(N-i)%MOD
    res %= MOD

for i in range(1,M):
    res += i*N*N*(M-i)%MOD
    res %= MOD

res *= comb(N*M-2,K-2)
res %= MOD
print(res)