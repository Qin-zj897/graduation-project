def solve(a):
    b=max(a)
    c=min(a)
    d=[]
    for x in a:
        if x!=b and x!=c:
            d.append(x)
    return d


if __name__ == '__main__':
    a = eval(input())
    result = solve(a)
    print(result)
