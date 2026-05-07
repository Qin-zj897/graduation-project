def init():
    global factorial
    MOD = 10 ** 9 + 7
    factorial = [0] * 200001
    factorial[0] = 1
    for i in range(1, 200001):
        factorial[i] = (factorial[i - 1] * i) % MOD

def main():
    init()
    n, m, k = map(int, input().split())
    s = 0
    MOD = 10 ** 9 + 7
    for i in range(n):
        s += (((((n - 1 - i) * (n - i)) // 2) % MOD) * ((m * m) % MOD)) % MOD
    for i in range(m):
        s += (((((m - 1 - i) * (m - i)) // 2) % MOD) * ((n * n) % MOD)) % MOD
    print((((s * ((factorial[n * m - 2] * pow(factorial[k - 2], MOD - 2, MOD)) % MOD)) % MOD) * pow(factorial[n * m - k], MOD - 2, MOD)) % MOD)

main()