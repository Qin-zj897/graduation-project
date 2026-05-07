import math,string,itertools,fractions,heapq,collections,re,array,bisect,sys,random,time,copy,functools
from collections import deque

sys.setrecursionlimit(10**7)
inf = 10**20
mod = 10**9 + 7

DR = [1, -1, 0, 0]
DC = [0, 0, 1, -1]

def LI(): return [int(x) for x in sys.stdin.readline().split()]
def LI_(): return [int(x)-1 for x in sys.stdin.readline().split()]
def LF(): return [float(x) for x in sys.stdin.readline().split()]
def LS(): return sys.stdin.readline().split()
def I(): return int(sys.stdin.readline())
def F(): return float(sys.stdin.readline())
def S(): return input()
     
# 10 ** 7 might cause MLE
fac = [-1] * (10**6+1)
inv = [-1] * (10**6+1)
finv = [-1] * (10**6+1)

fac[0] = fac[1] = 1
inv[1] = 1
finv[0] = finv[1] = 1

def initNCMMod(limit):
    for i in range(2, limit):
        fac[i] = fac[i-1] * i % mod
        inv[i] = mod - inv[mod%i] * (mod // i) % mod
        finv[i] = finv[i-1] * inv[i] % mod

def NCMMod(n, k):
    if n < k:
        return 0
    if (n < 0 or k < 0):
        return 0
    return fac[n] * (finv[k] * finv[n-k] % mod) % mod
initNCMMod(10**6)

def checkf(R, C):
    ans = 0
    for r1 in range(R):
        for c1 in range(C):
            for r2 in range(R):
                for c2 in range(C):
                    ans += abs(r1-r2) + abs(c1-c2)
    return ans


def main():
    R, C, K = LI()
    def f(i):
        return (i * (i+1) // 2) % mod

    ans = 0
    for i in range(C):
        ans += f(C-i-1) * R * R % mod
        ans += f(i) * R * R % mod

    for i in range(R):
        ans += f(R-i-1) * C * C % mod
        ans += f(i) * C * C % mod
    ans *= inv[2]
    ans %= mod
    ans *= NCMMod(R*C-2, K-2)
    ans %= mod

    print(ans)

main()

