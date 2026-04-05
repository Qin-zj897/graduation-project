def solve(a, b):
    c=[]
    c.append(a)
    for i in range(1,b):
        d=a*2*(0.5)**i
        c.append(d)
    d=sum(c)
    return "%.2f"%d


if __name__ == '__main__':
    a = eval(input())
    b = eval(input())
    result = solve(a, b)
    print(result)
