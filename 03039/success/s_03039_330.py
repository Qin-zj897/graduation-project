import numpy as np
MOD = 1000000007

def mod_add(a, b):
  return (a+b)%MOD

def mod_sub(a, b):
  return (a+MOD-b)%MOD

def mod_mul(a, b):
  return a*b%MOD

#互いに素なa,bについて、a*x+b*y=1の一つの解
def extgcd(a,b):
    r = (1,0,a)
    w = (0,1,b)
    while w[2]!=1:
        q = r[2]//w[2]
        r2 = w
        w2 = [r[0]-q*w[0],r[1]-q*w[1],r[2]-q*w[2]]
        r = r2
        w = w2
    return (w[0],w[1])

def mod_inv(a):
    x,  = extgcd(a, MOD)
    return (MOD+x%MOD)%MOD

def mod_div(a, b):
    return a*mod_inv(b)%MOD

def mod_pow(a, b):
    res = 1
    while b > 0:
        if b & 1:
            res = res * a * MOD
        a = a * a % MOD
        b >>= 1
    return res

MAX = 510000

fac  = [1,1]
finv = [1,1]
inv  = [1,1]

def COMinit():
    for i in range(2, MAX):
        fac.append(fac[i - 1] * i % MOD)
        inv.append(MOD - inv[MOD%i] * (MOD // i) % MOD)
        finv.append(finv[i - 1] * inv[i] % MOD)

def mod_COM(n, k):
    if n < k:
        return 0
    if n < 0 or k < 0:
        return 0
    return fac[n] * (finv[k] * finv[n - k] % MOD) % MOD

N, M, K = map(int, input().split())

res = 0
COMinit()
for d in range(1, N):
    tmp = mod_COM(M*N-2, K-2)
    tmp = mod_mul(tmp, d)
    tmp = mod_mul(tmp, (N-d)*M*M)
    res = mod_add(res, tmp)

for d in range(1, M):
    tmp = mod_COM(M*N-2, K-2)

    tmp = mod_mul(tmp, d)
    tmp = mod_mul(tmp, (M-d)*N*N)
    res = mod_add(res, tmp)


print(res)
