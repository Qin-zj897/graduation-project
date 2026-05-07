N, M, K = map(int, input().split())

MOD = 10**9 + 7
R = 1
for i in range(K - 2):
    R = (R * (N * M - 2 - i) * pow(i + 1, MOD - 2, MOD)) % MOD

ans = 0
for i in range(1, N + 1):
    for j in range(1, M + 1):
        u = (N - i + 1)
        d = i
        l = j
        r = (M - j + 1)
        D = 0
        D += u * l * (u + l - 2) // 2
        D += u * r * (u + r - 2) // 2
        D += d * l * (d + l - 2) // 2
        D += d * r * (d + r - 2) // 2
        D -= u * (u - 1) // 2
        D -= d * (d - 1) // 2
        D -= l * (l - 1) // 2
        D -= r * (r - 1) // 2
        ans += D

ans = (ans * R * pow(2, MOD - 2, MOD)) % MOD
print(ans)
