import math
H,W,K = map(int,input().split())

MOD = 10**9 + 7
class comb(object):
    def __init__(self, N, mod):
        self.mod = mod
        self.fac = [1] * N
        for i in range(1, N):
            self.fac[i] = (self.fac[i - 1] * i) % mod
        self.inv = [1] * N
        self.inv[N - 1] = pow(self.fac[N - 1], mod - 2, mod)
        for i in range(N - 1, 0, -1):
            self.inv[i - 1] = (self.inv[i] * i) % mod
 
    def calc(self, n, k):
        return self.fac[n] * (self.inv[k] * self.inv[n - k] %
                              self.mod) % self.mod

cb = comb(H*W, MOD).calc(H*W-2, K-2)

c = 0
for d in range(1,W):
    # score * 
    c += d*(W-d)*(H**2) * cb


for d in range(1,H):
    c += d*(H-d)*(W**2) * cb

print(c)