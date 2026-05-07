from operator import mul
from functools import reduce

def cmb(n,r):
    r = min(n-r,r)
    if r == 0: return 1
    over = reduce(mul, range(n, n - r, -1))
    under = reduce(mul, range(1,r + 1))
    return over // under
    
N, M, K = map(int, input().split())

ans = 0

for i in range(1, M):
    ans += (i * (M-i) * N**2) % (10**9 + 7) 
for i in range(1, N):
    ans += (i * (N-i) * M**2) % (10**9 + 7)

ans *= cmb((N*M-2)%(10**9+7) , K-2%(10**9+7))

print(int(ans % (10**9+7)))
