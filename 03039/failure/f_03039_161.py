MOD = 10 ** 9 + 7
n, m, k = map(int, input().split())
def f(i, j):
    return i*(i-1)*(i+1)//6 * j ** 2
def c(a, b):
    r = 1
    for i in range(a, a - b, -1):
        r = r * i // (a + 1 - i)
        r %= MOD
    return r
print((f(n, m) + f(m, n)) * c(n*m - 2, k - 2) % MOD)
