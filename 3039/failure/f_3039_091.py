def solve(t):
    p=t
    t.sort()
    a=t[0]
    b=t[-1]
    for i in p:
        if i==a:
            p.remove(a)
    for i in p:
        if i==b:
            p.remove(b)
    return p


if __name__ == '__main__':
    t = eval(input())
    result = solve(t)
    print(result)
