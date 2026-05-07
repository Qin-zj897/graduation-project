inpl = lambda: list(map(int,input().split()))
N, M ,K = inpl()
MOD = 10**9+7

def mod_inv(a, m=MOD):
    def mod_inv_sub(a, m):
        if a == 1:
            return 1, 0
        else:
            d, r = m//a, m % a
            x, y = mod_inv_sub(r, a)
            return y - d*x, x

    x = mod_inv_sub(a,m)[0]
    if x < 0:
        x += (-x//m + 1) * m
    return x

factor = 1
# MN-2 C K-2 = MN-2 C MN-K = MN-2 C L
L = min(K-2,M*N-K)
for i in range(M*N-L-1,M*N-1):
    factor = factor * i % MOD
for i in range(1,L+1):
    factor = factor * mod_inv(i) % MOD
ans = ((M*(N-1)*(N+1) + N*(M-1)*(M+1))*M*N //6 * factor) % MOD
print(ans)