def solve(h, N):
    if N == 1:
        return "%.2f"%h
    else:
        for i in range(N-1):
            h += h
        return "%.2f"%h


if __name__ == '__main__':
    h = float(input())
    N = int(input())
    result = solve(h, N)
    print(result)
