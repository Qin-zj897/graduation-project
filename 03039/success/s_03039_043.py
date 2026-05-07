mod = 10**9 + 7


def comb(n, k):
    x, y = 1, 1
    k = min(k, n - k)
    for i in range(k):
        x *= n - i
        x %= mod
        y *= i + 1
        y %= mod
    return x * pow(y, mod - 2, mod) % mod


def solve(string):
    n, m, k = map(int, string.split())
    base = n * m * (n + m) * (n * m - 1) // 6 % mod
    return str(base * comb(n * m - 2, k - 2) % mod)


if __name__ == '__main__':
    print(solve(input()))
