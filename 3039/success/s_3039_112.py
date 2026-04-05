def solve(a):
    b=max(a)
    c=min(a)
    d=[]
    for i in a:
        if i!=b and i!=c:
            d.append(i)
    return d


if __name__ == '__main__':
    a = eval(input())
    result = solve(a)
    print(result)
