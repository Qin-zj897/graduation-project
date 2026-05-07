n, m, k = map(int, input().split())

mod = 10**9+7
N = 10**5+50
fac = [1]*(N+1)
finv = [1]*(N+1)
for i in range(N):
    fac[i+1] = fac[i] * (i+1) % mod
finv[-1] = pow(fac[-1], mod-2, mod)
for i in reversed(range(N)):
    finv[i] = finv[i+1] * (i+1) % mod

def cmb1(n, r, mod):
    if r <0 or r > n:
        return 0
    r = min(r, n-r)
    return fac[n] * finv[r] * finv[n-r] % mod

ans = 0
for i in range(n*m-1):
    xi, yi = divmod(i, m)
    for j in range(i, n*m):
        xj, yj = divmod(j, m)
        d = abs(xi-xj)+abs(yi-yj)
        ans += d
        #ans += d*cmb1(n*m-2, k-2, mod)
ans *= cmb1(n*m-2, k-2, mod)
ans %= mod
print(ans)
