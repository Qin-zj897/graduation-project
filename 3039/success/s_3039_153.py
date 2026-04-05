def solve(a):
    c =max(a)
    b = min(a)
    if b ==c:
        return "[]"
    else:


        d = a.count(c)
        e = a.count(b)
        for i in range(1,d+1):
            a.remove(c)
        for i in range(1,e+1):
            a.remove(b)
        return a


if __name__ == '__main__':
    a = eval(input())
    result = solve(a)
    print(result)
