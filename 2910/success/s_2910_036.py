def solve(h, N):
    a=h
    s=h
    if N==1:
        s=h
    else:
        for i in range(N-1):
            h=h/2
            s=s+2*h
    return "%.2f"%s


if __name__ == '__main__':
    h = eval(input())
    N = eval(input())
    result = solve(h, N)
    print(result)
