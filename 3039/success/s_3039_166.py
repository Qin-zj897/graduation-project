def solve(a):
    b=[]
    c=max(a)
    d=min(a)
    e=[]
    for i in a:
        if i==d or i==c:
            e.append(i)
    for x in e:
        a.remove(x)
    for y in a:
        b.append(y)
    return b


if __name__ == '__main__':
    a = eval(input())
    result = solve(a)
    print(result)
