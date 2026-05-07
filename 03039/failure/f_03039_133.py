MOD = 10**9+7
MAX = 2*10**5
n, m, k = map(int, input().split())

fact, finv = [1]*(MAX+1), [1]*(MAX+1)
for i in range(1, MAX+1):
    fact[i] = (fact[i-1]*i)%MOD
    finv[i] = pow(fact[i], MOD-2, MOD)

def comb(n, r):
    return ((fact[n]*finv[n-k])%MOD*finv[k])%MOD

cumsum = [0]*(MAX+1)
for i in range(1, max(n, m)+1):
    cumsum[i] = cumsum[i-1]+i

ans = 0
for i in range(n):
    ans += ((cumsum[i] + cumsum[n-i-1])*n%MOD)*n%MOD
for i in range(m):
    ans += ((cumsum[i] + cumsum[m-i-1])*m%MOD)*m%MOD

ans //= 2
ans = (ans*comb(n*m-2, k-2))%MOD
print(ans)
