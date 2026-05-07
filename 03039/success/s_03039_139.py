class PERM_COMB_MOD():
    # http://drken1215.hatenablog.com/entry/2018/06/08/210000
    def __init__(self, max_n=510000, mod=10**9+7):
        self.fac = [0]*max_n
        self.finv = [0]*max_n
        self.inv = [0]*max_n
        self.fac[0] = self.fac[1] = 1
        self.finv[0] = self.finv[1] = 1
        self.inv[1] = 1
        self.max = max_n
        self.mod = mod
        self._maesyori()

    def _maesyori(self):
        for i in range(2,self.max):
            self.fac[i] = self.fac[i-1] * i % self.mod
            self.inv[i] = self.mod - self.inv[self.mod % i] * (self.mod // i) % self.mod
            self.finv[i] = self.finv[i-1] * self.inv[i] % self.mod

    def perm(self, n, k):
        if n < k : return 0
        if n < 0 or k < 0:return 0
        return self.fac[n] * self.finv[n-k] % self.mod % self.mod

    def comb(self, n, k):
        if n < k : return 0
        if n < 0 or k < 0:return 0
        return self.fac[n] * (self.finv[k] * self.finv[n-k] % self.mod) % self.mod

n,m,k = map(int, input().split())
mod = 10**9+7
p = PERM_COMB_MOD()
ans = 0

# xについて
for i in range(1,n):
    ans += i*(n-i)*m*m % mod
    ans %= mod
# yについて
for i in range(1,m):
    ans += i*(m-i)*n*n % mod
    ans %= mod
ans *= p.comb(n*m-2,k-2)
ans %= mod
print(ans)