def solve(h, n):
    a=h
    list1=[]
    for i in range(1,n):
        a=a/2
        s=2*a
        list1.append(s)
    list1.append(h)
    lc=sum(list1)
    return "%.2f"%(lc)


if __name__ == '__main__':
    h = eval(input())
    n = eval(input())
    result = solve(h, n)
    print(result)
