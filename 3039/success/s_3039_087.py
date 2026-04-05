def solve(a):
    b=max(a)
    c=min(a)
    d=a.copy()
    for x in d:
        if x==b:
            a.remove(b)
        elif x==c:
            a.remove(c)
    return a


if __name__ == '__main__':
    a = eval(input())
    result = solve(a)
    print(result)
