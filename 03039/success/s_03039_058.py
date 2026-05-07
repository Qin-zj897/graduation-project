N, M, K = map(int, input().split())

MOD = 10**9 + 7
L = N*M

fact = [1]*(L+1)
rfact = [1]*(L+1)
r = 1
for i in range(1, L+1):
  fact[i] = r = r * i % MOD
rfact[L] = r = pow(fact[L], MOD-2, MOD)
for i in range(L, 0, -1):
  rfact[i-1] = r = r * i % MOD

L = max(N, M)
D = [0]*(L+1)
for i in range(L):
    D[i+1] = D[i] + (i+1)

ans = 0
for i in range(N):
    for j in range(M):
        ans += D[j] * (i+1) + D[M-1-j] * i + D[i] * M
    ans %= MOD
print(ans * fact[N*M-2] * rfact[N*M-K] * rfact[K-2] % MOD)