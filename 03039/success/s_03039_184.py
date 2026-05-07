# machigatta 
class cmbs(object):
    def __init__(self, mod):
        self.mod = mod
        self.g1 = [1, 1]
        self.g2 = [1, 1]
        self.inverse = [0, 1]
        for i in range(2, 10 ** 6 + 1):
            self.g1.append((self.g1[-1] * i) % mod)
            self.inverse.append((-self.inverse[mod % i] * (mod // i)) % mod)
            self.g2.append((self.g2[-1] * self.inverse[-1]) % mod)

    def cmb(self, n, r):
        if n > 10 ** 6:
            return self.cmbl(n, r)
        return self.cmbr(n, r)

    def cmbr(self, n, r):
        if r < 0 or r > n:
            return 0
        r = min(r, n - r)
        return self.g1[n] * self.g2[r] * self.g2[n-r] % self.mod

    def cmbl(self, n, r):
        t = 1
        r = min(r, n-r)
        for i in range(n - r + 1, n + 1):
            t = t * i % self.mod
        return t * self.g2[r] % self.mod

def main():
    N, M, K = map(int, input().split())
    mod = 10**9 + 7
    c = cmbs(mod)
    r = c.cmb((N*M)-2, K-2)
    r2 = 0
    for i in range(N):
        r2 += ((i + 1) * i + (N - i - 1) * (N-i))
    r2 *= pow(M, 2, mod)
    r2 %= mod
    r3 = 0
    for i in range(M):
        r3 += ((i + 1) * i + (M - i - 1) * (M-i))
    r3 *= pow(N, 2, mod)
    r3 %= mod
    r = r * (r2 + r3) * c.inverse[4]#// 2
    return r % mod
print(main())
