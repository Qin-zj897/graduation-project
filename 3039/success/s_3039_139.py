def solve(a):
    b=max(a)
    c=min(a)
    d=a.count(b)
    e=a.count(c)
    if b>c:
        for i in range(d):
            a.remove(b)
        for i in range(e):
            a.remove(c)
    else:
        a=[]
    return a


if __name__ == '__main__':
    a = eval(input())
    result = solve(a)
    print(result)
