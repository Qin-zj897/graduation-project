N,M,K = map(int, input().split())

comb = 1
mod = 10**9+7
for i in range(min(K-2, N*M-K)):
    comb *= (N*M-2-i)
    comb //= i+1

comb %= mod
x = (N*N*(N+1)//2-N*(N+1)*(2*N+1)//6)*M**2
y = (M*M*(M+1)//2-M*(M+1)*(2*M+1)//6)*N**2
ans = (x+y)*comb
print(ans%mod)