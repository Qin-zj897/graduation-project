n, m, k = map(int, input().split())

MOD = 10 ** 9 + 7

class ModCmb:
    def __init__(self, size):
        self.inv = [1] * (size + 1)
        self.fact = [1] * (size + 1)
        temp_inv = [1] * (size + 1)
        for i in range(2, size + 1):
            temp_inv[i] = ( -(MOD // i) * temp_inv[MOD%i] ) % MOD
        for i in range(2, size + 1):
            self.fact[i] = self.fact[i-1] * i % MOD
            self.inv[i] = self.inv[i-1] * temp_inv[i] % MOD

    def cmb(self, n, r):
        if n <= 1 or n == r:
            return 1
        elif r == 1:
            return n
        else:
            return  (self.fact[n] * self.inv[r] % MOD ) * self.inv[n-r] % MOD

mc = ModCmb(n*m)

temp = mc.cmb(n*m-2, k-2)

ans = 0

for h in range(n):
    for w in range(m):
        if h != 0 and w != 0:
            ans += (h + w) * temp * (n - h) * (m - w) * 2
        else:
            ans += (h + w) * temp * (n - h) * (m - w)
        ans %= MOD

print(ans)