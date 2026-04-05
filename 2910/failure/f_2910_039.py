def solve(h, N):
    for x in range(N):
        h = h+0.5*h
    return '{:.2f}'.format(h)


if __name__ == '__main__':
    h = eval(input())
    N = eval(input())
    result = solve(h, N)
    print(result)
