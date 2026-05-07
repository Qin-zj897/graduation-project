N,M,K = map(int, input().split())

mod = 10**9+7
a = 1
b = 1
for i in range(min(K-2, N*M-K)):
    a *= (N*M-2-i)
    b *= i+1

comb = a//b
ans = ((N*N*(N+1)//2-N*(N+1)*(2*N+1)//6)*M**2 + (M*M*(M+1)//2-M*(M+1)*(2*M+1)//6)*N**2)%mod
ans *= comb
print(ans%mod)