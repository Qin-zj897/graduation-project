from math import factorial

MOD = 10 ** 9 + 7
N, M, K = map(int, input().split(' '))

n, r = N * M - 2, K - 2
#P = factorial(n) // factorial(n-r) // factorial(r) % MOD
P = factorial(n) * pow(factorial(n-r) % MOD, MOD - 2, MOD) * pow(factorial(r) % MOD, MOD - 2, MOD)

ans = 0
for n in range(N) :
    for m in range(M) :
        if n == 0 and m == 0 :
            continue

        d = n + m
        cnt = (N - n) * (M - m) * (1 if n == 0 or m == 0 else 2)

        ans += d * cnt

ans = (ans * P) % MOD
print(ans)
