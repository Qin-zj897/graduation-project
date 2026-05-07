MOD = 10 ** 9 + 7

def mod_combination(n, k):
    def extended_gcd(a, b):
        if b == 0:
            return (a, 1, 0)
        d, x, y = extended_gcd(b, a % b)
        return (d, y, x - (a // b) * y)
    p, q = 1, 1
    for i in range(n - k + 1, n + 1):
        p = (p * i) % MOD
    for i in range(2, k + 1):
        q = (q * i) % MOD
    return p * (extended_gcd(q, MOD)[1] % MOD) % MOD

def mod_pow(a, b):
    return pow(a, b, MOD)

def mod_inv(a):
    return pow(a, MOD - 2, MOD)


N, M, K = map(int, input().split())

X = mod_pow(M, 2) * (mod_pow(N, 3) - N) % MOD
Y = mod_pow(N, 2) * (mod_pow(M, 3) - M) % MOD

comb = mod_combination(N * M - 2, K - 2)

ans = (X + Y) * mod_inv(6) % MOD * comb % MOD

print(ans)
