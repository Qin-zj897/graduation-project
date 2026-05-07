N, M, K = map(int, input().split())

MOD = 10 ** 9 + 7

def combination(n, m):
    m = min(m, n-m)
    c = 1
    for i in range(m):
        c *= (n-i) * pow(m-i, MOD-2, MOD) % MOD
    return c % MOD

pattern = combination(N*M-2, K-2) % MOD

def diff_x_pattern(val):
    # for d in range(1, val):
    #     ret += (val - d) * d
    return (val * (val - 1) * val // 2 - (val - 1) * (2 * val - 1) * val // 6) % MOD

ans = (M * M * diff_x_pattern(N)) % MOD
ans += (N * N * diff_x_pattern(M)) % MOD
print((ans*pattern)%MOD)
