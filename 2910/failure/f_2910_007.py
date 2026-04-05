def solve(a, b):
    c=[]
    c.append(a)
    for i in range(1,b):
        a=a*(0.5)**(i-1)
        c.append(a)
    d=sum(c)
    return "%.2f"%d


if __name__ == '__main__':
    a = eval(input())
    b = eval(input())
    result = solve(a, b)
    print(result)
