def solve(h, N):
    sums=h
    for x in range(N-1):
        sums+=h*0.5**x
    return "%.2f"%(sums)


if __name__ == '__main__':
    h = int(input())
    N = int(input())
    result = solve(h, N)
    print(result)
