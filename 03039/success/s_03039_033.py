class FactMod:

    def __init__(self, n, mod):
        self.mod = mod
        self.f = [1] * (n + 1)
        for i in range(1, n + 1):
            self.f[i] = self.f[i - 1] * i % mod

        self.inv = [pow(self.f[-1], mod - 2, mod)]
        for i in reversed(range(1, n + 1)):
            self.inv.append(self.inv[-1] * i % mod)
        self.inv.reverse()

    def fact(self, n):
        """
        :return: n!
        """
        return self.f[n]

    def comb(self, n, r):
        """
        :return: nCr
        """
        return self.f[n] * self.inv[n - r] * self.inv[r] % self.mod


N, M, K = map(int, input().split())
mod = 10 ** 9 + 7

NM = N * M

ans = 0
for i in range(1, N):
    ans += (i * M * M * (N - i)) % mod
    ans %= mod

for i in range(1, M):
    ans += (i * N * N * (M - i)) % mod
    ans %= mod

factmod = FactMod(NM - 2, mod)
ans *= factmod.comb(NM - 2, K - 2)
ans %= mod
print(ans)
