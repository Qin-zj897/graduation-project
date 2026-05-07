# encoding:utf-8
import copy
import random
import bisect #bisect_left　これで二部探索の大小検索が行える
import fractions #最小公倍数などはこっち
import math
import sys

mod = 10**9+7
sys.setrecursionlimit(mod) # 再帰回数上限はでdefault1000

N,M,K = map(int,input().split())

def mCn(m,n):
    return math.factorial(m)//(math.factorial(n)*math.factorial(m-n)) % mod
#modに対応して高速なコンビネーションが求められる
# 階乗 & 逆元計算
n = 10**6
factorial = [1]
inverse = [1]
for i in range(1, n+2):
    factorial.append(factorial[-1] * i % mod)
    inverse.append(pow(factorial[-1], mod-2, mod))

def combinations_count(n,r):
    if n-r < 0:
        return 0
    return factorial[n]*inverse[r]*inverse[n-r]%mod


ans = 0
for i in range(1,M):
    ans += (M-i)*N**2*i*(combinations_count(M*N-2,K-2))
    if ans > mod:
        ans = ans % mod

for i in range(1,N):
    ans += (N-i)*M**2*i*(combinations_count(M*N-2,K-2))
    if ans > mod:
        ans = ans % mod

print(ans)
