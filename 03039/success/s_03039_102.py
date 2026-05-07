LARGE = 10**9 + 7


def p_inv(k, p):
    return pow(k, p-2, p)


def nck_mod(n, k, m):
    res = 1
    for i in range(n, n-k, -1):
        res = res * i % m
    for i in range(1, k+1):
        res = res * p_inv(i, m) % m
    return res


def solve(n, m, k):
    res_1 = 0
    r = n * (n - 1) // 2
    res_1 += r
    for i in range(1, n):
        r += i
        r -= n - i
        res_1 += r
    res_2 = 0
    r = m * (m - 1) // 2
    res_2 += r
    for i in range(1, m):
        r += i
        r -= m - i
        res_2 += r
    res = (res_1 * m * m + res_2 * n * n) % LARGE

    weight = nck_mod(n * m - 1, k - 1, LARGE) * p_inv(n * m - 1, LARGE) * (k - 1) * p_inv(2, LARGE) % LARGE

    return res * weight % LARGE


if __name__ == "__main__":
    N, M, K = map(int, input().split())
    print(solve(N, M, K))
    # print(solve(2, 2, 2))
    # print(solve(4, 5, 4))
    # print(solve(100, 100, 5000))
