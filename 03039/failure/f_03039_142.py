N,M,K = map(int, input().split())

x = 0
y = 0
comb = 1
mod = 10**9+7
for i in range(min(K-2, N*M-K)):
    comb *= (N*M-2-i)
for i in range(min(K-2, N*M-K)):
    comb //= i+1
    
comb %= mod
for d in range(1, N):
    x += d*(N-d)*M**2
for d in range(1, M):
    y += d*(M-d)*N**2
    
ans = x+y
ans %= mod
    
print((ans*comb)%mod)