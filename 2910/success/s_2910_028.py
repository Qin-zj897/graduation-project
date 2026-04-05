def solve(h, N):
    l = h
    for i in range(N-1):
        h = 0.5*h
        b = 2*h
        l += b
    return "%.2f"%l


if __name__ == '__main__':
    h = eval(input())
    N = eval(input())
    result = solve(h, N)
    print(result)
