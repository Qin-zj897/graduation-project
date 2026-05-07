import math
def choose(n,k):
    return math.factorial(n)//(math.factorial(n-k)*math.factorial(k))
mod = 10**9 +7
N,M,K = map(int,input().split())
ans = 0
memo = choose(N*M-2,K-2)
memo %= mod
#まずはX軸について、総和を考える
d = 1
while d < M:
    ans += d*memo*(M-d)*N*N
    d += 1
    ans %= mod

#Y軸について、総和を考える
d = 1
while d < N:
    ans += d*memo*(N-d)*M*M
    d += 1
    ans %= mod

print(ans)
