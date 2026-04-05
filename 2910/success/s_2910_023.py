def solve(h, N):
    s=h
    if N==1:
        return "%.2f"%h
    else:
        for i in range(N-1):
            s=s+h
            h=h/2
        return "%.2f"%s


if __name__ == '__main__':
    h = eval(input())
    N = eval(input())
    result = solve(h, N)
    print(result)
