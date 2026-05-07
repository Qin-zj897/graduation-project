N, M, K = map(int, input().split())
mod = 10**9+7
ans = 0
def combination(a,b, mod):
    b = min(b, a-b)
    ans = 1
    for i in range(0,b):
        ans *= (a-i)/(i+1) % mod
    return int(ans)
for d in range(1,N):
    ans += d*(N-d)*M**2 % mod
for d in range(1,M):
    ans += d*(M-d)*N**2 % mod
ans *= combination(N*M-2, K-2, mod) % mod
print(ans)