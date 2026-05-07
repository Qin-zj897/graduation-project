N,M,K = map(int, input().split())

mod = 10**9+7
a = 1
b = 1
for i in range(min(K-2, N*M-K)):
    a *= (N*M-2-i)
    a %= mod
    b *= i+1
    b %= mod
comb = a * pow(b, mod - 2, mod) % mod

ans = (((N*N*(N+1)//2-N*(N+1)*(2*N+1)//6)*M**2 + (M*M*(M+1)//2-M*(M+1)*(2*M+1)//6)*N**2)*comb)%mod
print(ans)