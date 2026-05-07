def cmb(n, r, p):
  if (r < 0) or (n < r):
    return 0
  r = min(r, n - r)
  return fac[n]*finv[r]*finv[n-r]%p

n,m,K = map(int,input().split())
MOD = pow(10,9)+7
N = pow(10,6)
fac = [-1]*(N+1); fac[0] = 1; fac[1] = 1 #階乗
finv = [-1]*(N+1); finv[0] = 1; finv[1] = 1 #階乗の逆元
inv = [-1]*(N+1); inv[0] = 0; inv[1] = 1 #逆元
for i in range(2,N+1):
  fac[i] = fac[i-1]*i%MOD
  inv[i] = MOD - inv[MOD%i]*(MOD//i)%MOD
  finv[i] = finv[i-1]*inv[i]%MOD

factor = cmb(n*m-2,K-2,MOD)
X = factor*(m-1)*m*(m+1)//6*n*n%MOD
Y = factor*(n-1)*n*(n+1)//6*m*m%MOD
#print(X,Y)
ans = (X+Y)%MOD
print(ans)