def solve(h, N):
    s=h
    if N==1:
        s=h
    else:
        for i in range(1,N):
            s=s+((0.5)**i)*h*2
    return "%.2f"%s


if __name__ == '__main__':
    h = eval(input())
    N = eval(input())
    result = solve(h, N)
    print(result)
