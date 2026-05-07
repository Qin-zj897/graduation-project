import sys
#import numpy as np
from scipy.misc import comb

s2nn = lambda s: [int(c) for c in s.split(' ')]
ss2nn = lambda ss: [int(s) for s in ss]
ss2nnn = lambda ss: [s2nn(s) for s in ss]
i2s = lambda: sys.stdin.readline().rstrip()
i2n = lambda: int(i2s())
i2nn = lambda: s2nn(i2s())
ii2ss = lambda n: [sys.stdin.readline().rstrip() for _ in range(n)]
ii2nn = lambda n: ss2nn(ii2ss(n))
ii2nnn = lambda n: ss2nnn(ii2ss(n))

def main():
    N, M, K = i2nn()
    L = max(M, N)
    dp = [0] * L
    for i in range(1, L):
        dp[i] = dp[i-1] + i
    n = 0
    for i in range(N):
        n += (dp[i] + dp[N-1-i]) * M * M
    for i in range(M):
        n += (dp[i] + dp[M-1-i]) * N * N
    n = int(n / 2)
    r = comb(N*M-2, K-2, exact=True)
    n = n * r % (int(1e+9)+7)
    print(n)

main()
