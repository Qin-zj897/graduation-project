def solve(a):
    c =max(a)
    b = min(a)
    if b ==c:
        return "[]"
    else:


        d = a.count(c)
        e = a.count(b)
        for i in range(d):
            a.remove(c)
        for i in range(e):
            a.remove(b)
        return a


if __name__ == '__main__':
    a = eval(input())
    result = solve(a)
    print(result)
