N,M,K = map(int, input().split())

mod = 10**9 + 7
n = 2*10**5

fact = [1]*(n+1)
rfact = [1]*(n+1)
r = 1
for i in range(1, n+1):
  fact[i] = r = r * i % mod
rfact[n] = r = pow(fact[n], mod-2, mod)
for i in range(n, 0, -1):
  rfact[i-1] = r = r * i % mod

def comb(n, k):
  return fact[n] * rfact[k] * rfact[n-k] % mod

base = comb(N*M-2,K-2)
first = 0
for i in range(N):
  for j in range(M):
    first += (i+j)
dp = [0]*M
dp[0] = first
for i in range(1,M):
  dp[i] = dp[i-1] - (M-2*i) * N
SUM = sum(dp)
dp2 = [0]*N
dp2[0] = SUM
for i in range(1,N):
  dp2[i] = dp2[i-1] - (N-2*i) * (M ** 2)
SUM = sum(dp2)
ans = comb(N*M-2,K-2) * (SUM % mod) * pow(2,mod-2,mod)
ans %= mod
print(ans)
