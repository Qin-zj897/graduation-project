class Combination:
    def __init__(self, n_max=10**6, mod=10**9+7):
        # self._n_max = n_max
        self._fac, self._finv, self._inv = [0]*n_max, [0]*n_max, [0]*n_max
        self._fac[0], self._fac[1] = 1, 1
        self._finv[0], self._finv[1] = 1, 1
        self._inv[1] = 1
        for i in range(2, n_max):
            self._fac[i] = self._fac[i - 1] * i % MOD
            self._inv[i] = MOD - self._inv[MOD%i] * (MOD // i) % MOD
            self._finv[i] = self._finv[i - 1] * self._inv[i] % MOD
    def com(self, n, r):
        if n < r: return 0
        if n < 0 or r < 0: return 0
        return self._fac[n] * (self._finv[r] * self._finv[n - r] % MOD) % MOD


def modinv(a,m):
    b, u, v = m, 1, 0
    while b:
        t = a//b
        a -= t*b
        a,b = b,a
        u -= t * v
        u,v = v,u
    u %= m
    return u


MOD = 10**9+7
comb = Combination(10**6, MOD)
n,m,k = map(int, input().split())

ans = 0
for i in range(m):
    other_sum = i*(i+1)//2 + (m-1-i)*(m-i)//2
    # other_sum *= n
    pattern = n*n*comb.com(n*m-2,k-2)
    ans += pattern*other_sum
    ans%=MOD

for i in range(n):
    other_sum = i*(i+1)//2 + (n-1-i)*(n-i)//2
    # other_sum *= m
    pattern = m*m*comb.com(n*m-2,k-2)
    ans += pattern*other_sum
    ans%=MOD

a = modinv(2,MOD)
ans = (ans*a)%MOD
print(ans)