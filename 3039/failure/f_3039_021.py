def solve(a):
    b=max(a)
    c=min(a)
    for i in a:
        if i==b:
           a.remove(b)
    for t in a:
        if t==c:
            a.remove(c)
    return a


if __name__ == '__main__':
    a = eval(input())
    result = solve(a)
    print(result)
