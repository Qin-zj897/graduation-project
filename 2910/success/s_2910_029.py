def solve(h, n):
    s=h
    if n==1:
        return "%.2f"%(h)
    else:
        while n>0:
            s+=2*h*0.5
            h=h/2
            n-=1
    return "%.2f"%(s)


if __name__ == '__main__':
    h = float(input())
    n = int(input())-1
    result = solve(h, n)
    print(result)
