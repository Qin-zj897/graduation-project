import sys
import itertools
# import numpy as np
import time
import math
import heapq
from collections import defaultdict
from collections import Counter
sys.setrecursionlimit(10 ** 7)
 
INF = 10 ** 18
MOD = 10 ** 9 + 7
read = sys.stdin.buffer.read
readline = sys.stdin.buffer.readline
readlines = sys.stdin.buffer.readlines

# map(int, input().split())

N, M, K = map(int, input().split())
MAX = 2 * 10 ** 5 + 5
MOD = 10 ** 9 + 7

fac = [0 for i in range(MAX)]
finv = [0 for i in range(MAX)]
inv = [0 for i in range(MAX)]

def comInit(mod):
    fac[0], fac[1] = 1, 1
    finv[0], finv[1] = 1, 1
    inv[1] = 1
    for i in range(2, MAX):
        fac[i] = fac[i - 1] * i % mod
        inv[i] = mod - inv[mod % i] * (mod // i) % mod
        finv[i] = finv[i - 1] * inv[i] % mod


def com(n, r, mod):
    if n < r:
        return 0
    if n < 0 or r < 0:
        return 0
    return fac[n] * (finv[r] * finv[n - r] % mod) % mod

def p(n, r, mod):
    return fac[n] * finv[n - r] % mod
comInit(MOD)

xs = 0
for i in range(M):
    xs += i * (M - i)
    xs %= MOD
xs *= N ** 2
ys = 0
for i in range(N):
    ys += i * (N - i)
    ys %= MOD
ys *= M ** 2
ys %= MOD
print(((xs + ys) % MOD) * com(N * M - 2, K - 2, MOD) % MOD)