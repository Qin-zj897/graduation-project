n, m, k = map(int, input().split())
mod = 10**9 + 7

def powmod(x, n):
  ret = 1
  while n > 0:
    if n & 1:
      ret *= x; ret %= mod; n -= 1
    else:
      x *= x; x %= mod; n >>= 1
  return ret

fact = [1 for _ in range(200010)]
revfact = [1 for _ in range(200010)]

def setfact(n):
  for i in range(n):
    fact[i+1] = fact[i] * (i+1); fact[i+1] %= mod
  revfact[n] = powmod(fact[n], mod-2)
  for i in range(n):
    revfact[n-i-1] = revfact[n-i] * (n-i); revfact[i] %= mod
  return

def getC(n, r):
  return fact[n] * revfact[r] % mod * revfact[n-r] % mod
  
setfact(n*m)
ans = 0
for i in range(m):
    ans += n*n*(m-i)*i
    ans %= mod
for i in range(n):
    ans += m*m*(n-i)*i
    ans %= mod
print(ans * getC(n*m-2, k-2) % mod)