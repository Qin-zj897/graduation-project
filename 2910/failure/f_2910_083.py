def solve(h, n):
    c=[]
    for i in range (n-1):
        h=h*0.5
        c.append(h)
    d=sum(c)
    e=2*d
    f=e+h
    return "%.2f"%(f)


if __name__ == '__main__':
    h = eval(input())
    n = eval(input())
    result = solve(h, n)
    print(result)
