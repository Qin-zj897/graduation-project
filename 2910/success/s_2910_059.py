def solve(h, N):
    s = h
    l = 0
    for i in range(1,N+1):
        s = s + 2 * l
        l = h*(0.5**i)
    return "%.2f"%(s)


if __name__ == '__main__':
    h = eval(input())
    N = eval(input())
    result = solve(h, N)
    print(result)
