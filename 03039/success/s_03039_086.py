import math
N, M, K = map(int,input().split())
MOD = 10**9+7

def nCr(n,r):
    return (math.factorial(n)//(math.factorial(r)*math.factorial(n-r)))%MOD

ans = 0
t = nCr(N*M-2,K-2)
for d in range(1,M):
    ans += d*(M-d)*(N*N)*t
    ans %= MOD

for d in range(1,N):
    ans += d*(N-d)*(M*M)*t
    ans %= MOD

print(ans)