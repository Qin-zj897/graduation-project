N, M, K = map(int, input().split())

MOD = 10 ** 9 + 7

def combination(n, m):
    c = 1
    for i in range(m):
        c *= n-i
    for i in range(1, m+1):
        c //= i
    return c

pattern = combination(M*N-2, K-2) % MOD

ans = 0
for d in range(1, N):
    ans += M * M * (N - d) * d
for d in range(1, M):
    ans += N * N * (M - d) * d
print((ans*pattern)%MOD)
