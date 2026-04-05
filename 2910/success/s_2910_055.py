def solve(h, n):
    s=0
    if n==1:
        return "%.2f"%(h)
    else:
        s=h
        h=0.5*h
        for i in range(1,n):
            s=s+2*h
            h=0.5*h
        return "%.2f"%(s)


if __name__ == '__main__':
    h = eval(input())
    n = eval(input())
    result = solve(h, n)
    print(result)
