def solve(a):
    b=max(a)
    c=min(a)
    d=[]
    d.append(b)
    d.append(c)
    f=[]
    for x in a:
        if x not in d:
            f.append(x)
    return f


if __name__ == '__main__':
    a = eval(input())
    result = solve(a)
    print(result)
