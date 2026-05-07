n, m, k = map(int, input().split())

MOD = 1000000007
def pow_mod(x, y):
    if y == 0: return 1
    ans = 1
    while y > 1:
        if y % 2 != 0: 
            ans *= x
            ans %= MOD
        x *= x
        x %= MOD
        y //= 2
    return ans * x % MOD
mod_inv = lambda x: pow_mod(x, MOD - 2)

a = 0
for i in range(1, n):
  a += (i * (n - i)) % MOD 
  a %= MOD
a *= (m * m) % MOD
a %= MOD

b = 0
for i in range(1, m):
  b += (i * (m - i)) % MOD 
  b %= MOD
b *= (n * n) % MOD
b %= MOD

w = (m * n - 2) % MOD
wCx = {0: 1}
for i in range(1, k+1):
  wCx[i] = wCx[i - 1] 
  wCx[i] *= (w - i + 1) % MOD
  wCx[i] %= MOD
  wCx[i] *= mod_inv(i)
  wCx[i] %= MOD

ans = ((a + b) % MOD) 
ans *= wCx[k-2]
ans %= MOD

print(ans)