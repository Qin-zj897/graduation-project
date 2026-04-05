def solve(a):
    b=max(a)
    c=min(a)
    d=[]
    for x in a:
        if x==b:
            d.append(b)
        elif x==c:
            d.append(c)
        else:
            continue
    for i in d:
        a.remove(i)
    return a


if __name__ == '__main__':
    a = eval(input())
    result = solve(a)
    print(result)
