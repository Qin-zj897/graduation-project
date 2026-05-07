from math import factorial


def comb(n, k, mod):
  if n < 0 or k < 0 or n < k:
      return 0
  if n == 0 or k == 0:
      return 1
  
  a=factorial(n) % mod
  b=factorial(k) % mod
  c=factorial(n-k) % mod
  return (a*power_func(b, mod-2, mod)*power_func(c, mod-2, mod)) % mod


def power_func(a,b,mod):
  if b == 0:
      return 1
  if b % 2 == 0:
    d = power_func(a, b//2, mod)
    return d*d % mod
  if b % 2 == 1:
    return (a*power_func(a, b-1, mod)) % mod


N, M, K = map(int, input().split())
mod = 10**9 + 7


x = N*N*(M*M*(M+1)//2 - M*(M+1)*(2*M+1)//6)
y = M*M*(N*N*(N+1)//2 - N*(N+1)*(2*N+1)//6)
ans = comb(N*M-2, K-2, mod) * (x + y) % mod
print(ans)