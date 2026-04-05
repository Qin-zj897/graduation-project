def solve(t):
    t.sort()
    a=t[0]
    b=t[-1]
    for i in t:
        if i==a:
            t=t[1:]
        elif i==b:
            t=t[:-1]
    return t


if __name__ == '__main__':
    t = eval(input())
    result = solve(t)
    print(result)
