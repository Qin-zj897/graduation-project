N, M, K = map(int, input().split())
MOD = 10 ** 9 + 7

def pow(x, n) :
    y = 1
    while n > 0 :
        if n & 1 :
            y = (y * x) % MOD
        x = (x * x) % MOD
        n >>= 1

    return y

def comb(n, r) :
    if n < r or n < 0 :
        return 0

    factorial = [0] * (n + 1)
    factorial[0] = 1

    for i in range(1, n + 1) :
        factorial[i] = (factorial[i-1] * i) % MOD

    return (factorial[n] * pow(factorial[n-r], MOD - 2) * pow(factorial[r], MOD - 2)) % MOD

res = 0

for i in range(N) :
    for j in range(M) :
        if i > 0 and j > 0 :
            res += 2 * (N - i) * (M - j) * (i + j)
        else :
            res += (N - i) * (M - j) * (i + j)

res *= comb(N * M - 2, K - 2)

print(res % MOD)