def cmb(n, r):
    return fact[n] * inv_fact[r] * inv_fact[n - r] % MOD


MOD = 10 ** 9 + 7

N, M, K = map(int, input().split())

fact = [1]
for i in range(1, N * M - 2 + 1):
    fact.append(fact[-1] * i % MOD)

inv_fact = [-1] * (N * M - 1)
inv_fact[-1] = pow(fact[-1], MOD - 2, MOD)
for i in range(N * M - 3, -1, -1):
    inv_fact[i] = inv_fact[i + 1] * (i + 1) % MOD

pat = cmb(N * M - 2, K - 2)

ans = 0

for i in range(M):
    ans += i * (M - i) * N ** 2

for i in range(N):
    ans += i * (N - i) * M ** 2

ans = (ans * pat) % MOD

print(ans)
