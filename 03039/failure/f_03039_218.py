from scipy.misc import comb


def solve(string):
    n, m, k = map(int, string.split())
    p = 10**9 + 7
    base = n * m * ((n + 1) * (m + 1) * (n + m - 2) - n**2 - m**2 + 2) // 6 % p
    return str(base * (comb(n * m - 2, min(k - 2, n * m - k), exact=True) % p) % p)


if __name__ == '__main__':
    print(solve(input()))
