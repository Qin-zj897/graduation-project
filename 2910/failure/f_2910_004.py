def solve(a, b):
    c=[]
    for i in range(b):
        a=a*(1/2)
        c.append(a)
    d=sum(c)
    return "%.2f"%c


if __name__ == '__main__':
    a = eval(input())
    b = eval(input())
    result = solve(a, b)
    print(result)
