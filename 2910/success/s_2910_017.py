def solve(h, N):
    num1 = 0
    if N == 1:
        num1 = h
    else:
        num1 += h
        for x in range(2, N + 1):
            H = h * 0.5 ** (x - 1)
            num1 += H * 2
    return "%.2f" % num1


if __name__ == '__main__':
    h = eval(input())
    N = eval(input())
    result = solve(h, N)
    print(result)
