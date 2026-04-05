def solve(h, n):
    total_h = h
    for i in range(n-1):
        total_h += h*(0.5)**i
    return "%.2f"%total_h


if __name__ == '__main__':
    h = int(input())
    n = int(input())
    result = solve(h, n)
    print(result)
