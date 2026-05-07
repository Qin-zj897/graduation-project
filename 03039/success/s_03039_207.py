from itertools import*
import math
from collections import*
from heapq import*
from bisect import bisect_left,bisect_right
from copy import deepcopy
inf = float("inf")
mod = 10**9+7
from functools import reduce
import sys
sys.setrecursionlimit(10**7)

n,m,k = map(int,input().split())
Max = n*m
#二項係数とその逆元テーブルを作る前処理
fac = [0]*(Max)
finv = [0]*(Max)
inv = [0]*(Max)
fac[0]=fac[1]=1
finv[0]=finv[1]=1
inv[1]=1
for i in range(2,Max):
        fac[i] = fac[i-1] * i % mod
        inv[i] = mod - inv[mod%i]*(mod//i)%mod
        finv[i] = finv[i-1]*inv[i]%mod
#O(1)でmod計算した組合せ数を計算
def C(n,r):
    if n < r:
        return 0
    if n < 0 or r < 0 :
        return 0
    return fac[n]*(finv[r]*finv[n-r]%mod)%mod
#mod計算した順列の計算
def P(n,r):
    if n < r:
        return 0
    if n<0 or r<0:
        return 0
    return (fac[n]*finv[n-r])%mod


c = C(n*m-2,k-2)
res = 0

for i in range(1,n):
    res += i * (n-i) *(m**2)*c
    res %= mod

for i in range(1,m):
    res += i*(m-i)*(n**2)*c
    res %= mod
print(res)