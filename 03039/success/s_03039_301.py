N,M,K = map(int,input().split())

mod = 10**9+7
fact = [1]*(2*N*M+1)
for n in range(1,2*N*M+1):
    fact[n] = n*fact[n-1]%mod

def inv(n):
    n %= mod
    return pow(n,mod-2,mod)

ans = M*(M**2-1)*N**2%mod+N*(N**2-1)*M**2%mod
ans *= inv(6)*fact[N*M-2]*inv(fact[K-2]*fact[N*M-K])
ans %= mod

print(ans)