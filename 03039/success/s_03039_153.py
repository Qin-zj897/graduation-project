import math

MOD = int(10**9 + 7)

def egcd(a, b):
    if a ==  0:
        return (b, 0, 1)
    else:
        g, y, x = egcd(b%a, a)
        return (g, x - (b//a) * y, y)

def modinv(a, m):
    g, x, y = egcd(a, m)
    if g != 1:
        raise Exception("No inverse exists")
    else:
        return x % m

def combination(n, r):
    return math.factorial(n)%MOD * modinv(math.factorial(r), MOD) * modinv(math.factorial(n-r), MOD) % MOD
    # return math.factorial(n) // math.factorial(r) // math.factorial(n-r)

N, M, K = map(lambda x: int(x), input().split())

ans = 0

k = combination(N*M-2, K-2)

for d in range(1, N):
    ans += (d * k * M * M * (N - d))%MOD

for d in range(1, M):
    ans += (d * k * N * N * (M - d))%MOD

print(ans % MOD)
