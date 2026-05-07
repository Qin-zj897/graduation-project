def solve(string):
    n, m, k = map(int, string.split())
    base = n * m * ((n + 1) * (m + 1) * (n + m - 2) - n**2 - m**2 + 2) // 6 % (10**9 + 7)
    return str(base)


if __name__ == '__main__':
    print(solve(input()))
