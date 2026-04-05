def solve(h, N):
    for i in range(N-1):
        h=h/2
    return "%.2f"%h


if __name__ == '__main__':
    h = eval(input())
    N = eval(input())
    result = solve(h, N)
    print(result)
