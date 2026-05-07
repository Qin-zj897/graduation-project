from functools import reduce

def combination2(n, r, MOD=10**9+7):
    if not 0 <= r <= n: return 0
    r = min(r, n - r)
    numerator = reduce(lambda x, y: x * y % MOD, range(n, n - r, -1), 1)
    denominator = reduce(lambda x, y: x * y % MOD, range(1, r + 1), 1)
    return numerator * pow(denominator, MOD - 2, MOD) % MOD

def q(x,y):
    return cn*(x**2)*(y**3-y)//6

N, M, K = map(int, input().split())
MOD=10**9+7
cn = combination2(N*M - 2, K-2, MOD)
print((q(N,M) + q(M,N))%MOD)
