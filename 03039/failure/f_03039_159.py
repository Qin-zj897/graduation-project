def combination(n, r):
    r = min(n - r, r)
    result = 1
    for i in range(n, n - r, -1):
        result *= i
    for i in range(1, r + 1):
        result //= i
    return result


N, M, K = map(int, input().split())
mod = 10 ** 9 + 7

NM = N * M
fa = [1] * (NM + 1)

for i in range(1, NM + 1):
    fa[i] = (fa[i - 1] * i) % mod

ans = 0
for i in range(1, N):
    ans += i * M * M * (N - i)
    ans %= mod

for i in range(1, M):
    ans += i * N * N *(M-i)
    ans %= mod

ans *= combination(NM-2, K-2)
ans %= mod
print(ans)
