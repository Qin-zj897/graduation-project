def solve(h, n):
    sums=h
    for x in range(n-1):
        sums+=h*(0.5)**(x)
    return "%.2f"%(sums)


if __name__ == '__main__':
    h = int(input())
    n = int(input())
    result = solve(h, n)
    print(result)
