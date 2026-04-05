def solve(h, N):
    s=h
    for x in range(N-1):
        h=h/2
        s=s+h*2
    return "%.2f"%s


if __name__ == '__main__':
    h = eval(input())
    N = eval(input())
    result = solve(h, N)
    print(result)
