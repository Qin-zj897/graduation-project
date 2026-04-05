def solve(h, N):
    s=h
    for i in range(N-1):
        h=s+h
        s=s*0.5
    return "%.2f"%(h)


if __name__ == '__main__':
    h = eval(input())
    N = eval(input())
    result = solve(h, N)
    print(result)
