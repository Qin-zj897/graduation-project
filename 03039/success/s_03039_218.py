MOD = 10**9 + 7

N, M, K = map(int, input().split())

invs = [1] * (K-1)
for x in range(2, K-1):
    invs[x] = (-(MOD//x) * invs[MOD%x]) % MOD

C = 1
for x in range(N*M-K+1, N*M-1):
    C *= x
    C %= MOD
for x in range(1, K-1):
    C *= invs[x]
    C %= MOD

ans1 = 0
for d in range(1, N):
    ans1 += (N-d) * d
    ans1 %= MOD
ans1 *= M**2
ans1 %= MOD

ans2 = 0
for d in range(1, M):
    ans2 += (M-d) * d
    ans2 %= MOD
ans2 *= N**2
ans2 %= MOD

print((ans1+ans2) * C % MOD)
