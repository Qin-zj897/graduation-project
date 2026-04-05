def solve(h, N):
    s = 0
    for x in range(N):
        s = s + h
        h = (1/2)*h
    return '{:.2f}'.format(s)


if __name__ == '__main__':
    h = eval(input())
    N = eval(input())
    result = solve(h, N)
    print(result)
