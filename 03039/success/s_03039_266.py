# -*- coding: utf-8 -*-
"""
Created on Wed May 29 22:49:19 2019

@author: Yamazaki Kenichi
"""

# -*- coding: utf-8 -*-
"""
Created on Sat May 25 21:24:29 2019

@author: Yamazaki Kenichi
"""

N, M, K = map(int,input().split())
mod = 10**9+7

def cmb(n, r, mod):
    if (r < 0 or r > n):
        return 0
    r = min(r, n-r)
    return g1[n] * g2[r] * g2[n-r] % mod

g1 = [1, 1] # 元テーブル
g2 = [1, 1] # 逆元テーブル
inv = [0, 1] # 逆元テーブル計算用テーブル

for i in range(2, N*M+1):
    g1.append((g1[-1] * i) % mod)
    inv.append((-inv[mod % i] * (mod//i)) % mod)
    g2.append((g2[-1] * inv[-1]) % mod)

ans1 = 0
def c(N,M):
    res = 0
    for i in range(1,N):
        res += i*(N-i)*M*M
    return res
ans1 += c(N,M) + c(M,N)
ans1 %= mod

ans1 *= cmb(N*M-2,K-2,mod)
ans1 %= mod

print(ans1)
