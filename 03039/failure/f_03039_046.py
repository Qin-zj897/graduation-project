N, M, K = map(int, input().split())
MOD = 10 ** 9 + 7

def comb(n, r) :
    if n - r < 0 or r < 0 :
        return 0
    
    r = min(n-r, r)

    a, b = 1, 1
    for i in range(r) :
        a *= n - i
        b *= i + 1

    return a // b

res = 0

for i in range(1, N + 1) :
    for j in range(1, M + 1) :
        if i > 1 and j > 1 :
            res += 2 * (N + 1 - i) * (M + 1 - j) * (i + j - 2)
        else :
            res += (N + 1 - i) * (M + 1 - j) * (i + j - 2)

res *= comb(N * M - 2, K - 2)

print(res % MOD)