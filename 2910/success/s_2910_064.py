def solve(h, n):
    s = h
    for i in range(n-1):
        h = h * (0.5)
        s = s + 2 * h
    return '%.2f'%s


if __name__ == '__main__':
    h = int(input())
    n = int(input())
    result = solve(h, n)
    print(result)
