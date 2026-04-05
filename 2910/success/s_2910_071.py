def solve(h, N):
    n = h
    for i in range(N-1):
        h += n
        n = n/2
    return "%.2f"%(h)


if __name__ == '__main__':
    h = int(input())
    N = int(input())
    result = solve(h, N)
    print(result)
