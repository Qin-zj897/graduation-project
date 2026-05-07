import math
n, m, k = map(int, input().split())

mod = 10**9 + 7

def mpow(x, n):
  ans = 1
  while n != 0:
    if n&1:
      ans = ans * x % mod
    x = x * x % mod
    n = n // 2
  return ans

fac = [0] * (n*m+2)
finv = [0] * (n*m+2)
inv = [0] * (n*m+2)
fac[0] = 1
fac[1] = 1
finv[0] = 1
finv[1] = 1
inv[1] = 1

for i in range(2, n*m+1):
  fac[i] = fac[i-1] * i % mod
  inv[i] = mod - inv[mod%i] * (mod // i) % mod
  finv[i] = finv[i-1] * inv[i] % mod

def choose(n,k):
  if n == 0 and k == 0:
    return 1
  if n < 0:
    return 0
  if n < k:
    return 0
  return fac[n] * (finv[k] * finv[n-k] % mod) % mod


sum = 0
for i in range(1, n+1):
  for j in range(1, m+1):
    score = i-1 + j-1
    cnt = (n-i+1) * (m-j+1)
    sum += (score * cnt) % mod
    #print((i,j), score, cnt)

for i in range(2, n+1):
  for j in range(1, m):
    score = i-1 + m-j
    cnt = (n-i+1) * j
    sum += (score * cnt) % mod
    #print((i,j), score, cnt)

sum = sum * choose(m*n-2,k-2)
sum = sum % mod
print(sum)

