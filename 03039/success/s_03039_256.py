# coding: utf-8
# Your code here!
# coding: utf-8
# Your code here!

import sys

n,m,k = [int(i) for i in input().split()]
MOD=10**9+7
SIZE = 2*(10**5)
inv = [0]*(SIZE+1)# inv[j] = j^{-1} mod MOD
fac = [0]*(SIZE+1)# fac[j] = j! mod MOD
finv = [0]*(SIZE+1)# finv[j] = (j!)^{-1} mod MOD
inv[1] = 1
fac[0] = fac[1] = 1
finv[0] = finv[1] = 1
for i in range(2,SIZE+1):
    inv[i] = MOD -(MOD//i)*inv[MOD%i]%MOD
    fac[i] = fac[i-1]*i%MOD
    finv[i]= finv[i-1]*inv[i]%MOD

def choose(n,r): # nCk mod MOD の計算
    if 0 <= r <= n:
        return (fac[n]*finv[r]%MOD)*finv[n-r]%MOD
    else:
        return 0


ans0 = 0
ans1 = 0
for i in range(n):
    ans0 += i*(n-i)
    ans0 %= MOD
ans0 *= m
ans0 %= MOD
ans0 *= m
ans0 %= MOD

for i in range(m):
    ans1 += i*(m-i)
    ans1 %= MOD
ans1 *= n
ans1 %= MOD
ans1 *= n
ans1 %= MOD

ans = (ans0+ans1)%MOD


print(ans*choose(n*m-2,k-2)%MOD)

