def solve(h, N):
    s=10
    if N==1:
        return h
    else:
        for x in range(N-1):
            h*=0.5
            s+=h*2
        return "%.2f"%s


if __name__ == '__main__':
    h = eval(input())
    N = eval(input())
    result = solve(h, N)
    print(result)
