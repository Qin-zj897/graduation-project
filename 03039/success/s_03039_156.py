def comb(n, k, mod=10**9 + 7):
    from functools import reduce
    from operator import mul
    k = min(k, n - k)

    def calc(x, y):
        return x * y % mod

    x = reduce(calc, range(n - k + 1, n + 1), 1) % mod
    y = reduce(calc, range(1, k + 1), 1) % mod
    return x * pow(y, mod - 2, mod) % mod


def solve(string):
    n, m, k = map(int, string.split())
    mod = 10**9 + 7
    base = n * m * (n + m) * (n * m - 1) // 6 % mod
    return str(base * comb(n * m - 2, k - 2) % mod)


if __name__ == '__main__':
    print(solve(input()))
