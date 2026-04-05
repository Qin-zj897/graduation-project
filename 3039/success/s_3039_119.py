def solve(a):
    b=max(a)
    c=min(a)
    m=[]
    m.append(b)
    m.append(c)
    f=[]
    for x in a:
        if x not in m:
            f.append(x)
    return f


if __name__ == '__main__':
    a = eval(input())
    result = solve(a)
    print(result)
