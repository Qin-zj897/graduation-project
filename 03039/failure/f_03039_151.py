from scipy.misc import comb


def solve(string):
    n, m, k = map(int, string.split())
    return str(n * m * ((n + 1) * (m + 1) * (n + m - 2) - n**2 - m**2 + 2) // 6 *
               comb(n * m - 2, k - 2, exact=True) % (10**9 + 7))


if __name__ == '__main__':
    print(solve(input()))
