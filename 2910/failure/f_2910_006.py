def solve(a, b):
    c=[]
    c.append(a)
    for i in range(b):
        a*=(1/2)**(i-1)
        c.append(a)
    d=sum(c)
    return "%.2f"%d


if __name__ == '__main__':
    a = eval(input())
    b = eval(input())
    result = solve(a, b)
    print(result)
