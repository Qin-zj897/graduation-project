def solve(h, N):
    if N==1:
        return "%.2f"%(h)
    while N>1:
        d=h
        for x in range(N-1):
            c=h/2
            d=d+2*c
            h=h/2
        break
    return "%.2f"%(d)


if __name__ == '__main__':
    h = eval(input())
    N = eval(input())
    result = solve(h, N)
    print(result)
