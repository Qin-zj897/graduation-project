from itertools import combinations
import math
def cmb(n, r, p):
    if (r < 0) or (n < r):
        return 0
    r = min(r, n - r)
    return fact[n] * factinv[r] * factinv[n-r] % p

p = 10 ** 9 + 7
N = 10 ** 6  # N は必要分だけ用意する
fact = [1, 1]  # fact[n] = (n! mod p)
factinv = [1, 1]  # factinv[n] = ((n!)^(-1) mod p)
inv = [0, 1]  # factinv 計算用
 
for i in range(2, N + 1):
    fact.append((fact[-1] * i) % p)
    inv.append((-inv[p % i] * (p // i)) % p)
    factinv.append((factinv[-1] * inv[-1]) % p)


N,M,K = map(int,input().split())
ans = 0
mod = 10**9 +7
by = cmb(N*M-2,K-2,mod)

#まずは、X軸の総和を求める。
X = []
for i in range(1,N+1):
    for j in range(M):
        X.append(i)

for a,b in list(combinations(X,2)):
    ans = (ans + abs(a-b)*by%mod)%mod


Y = []
for i in range(1,M+1):
    for j in range(N):
        Y.append(i)

for a,b in list(combinations(Y,2)):
    ans = (ans + abs(a-b)*by%mod)%mod

print(ans)