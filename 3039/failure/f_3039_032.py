def solve(a):
    b= a.count(max(a))
    c= a.count(min(a))
    for x in range(b):
        del a[a.index(max(a))]
    for y in range(c):
        del a[a.index(min(a))]
    return a


if __name__ == '__main__':
    a = eval(input())
    result = solve(a)
    print(result)
