def solve(a):
    b=max(a)
    c=min(a)
    d=a.copy()
    for x in a:
        if x==b or x==c:
            d.remove(x)
    return d


if __name__ == '__main__':
    a = eval(input())
    result = solve(a)
    print(result)
