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

def a(N,M):
    return N*(N+1)//2*(M+1) + M*(M+1)//2*(N+1)
def b(N):
    return N*(N+1)//2

ans = 0
for i in range(N):
    for j in range(M):
        ans += a(i,j) + a(N-i-1,j) + a(i,M-j-1) + a(N-i-1,M-j-1)
        ans -= b(i) + b(j) + b(N-i-1) + b(M-j-1)
        ans %= mod
#        print(i,j,a(i,j),a(N-i-1,j),a(i,M-j-1),a(N-i-1,M-j-1),b(i),b(j),ans)
ans *= g2[2]

ans *= cmb(N*M-2,K-2,mod)
ans %= mod

print(ans)
