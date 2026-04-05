def solve(l):
    a=max(l)
    b=min(l)
    for i in range(l.count(a)):
        l.remove(a)
    for i in range(l.count(b)):
        l.remove(b)
    return l


if __name__ == '__main__':
    l = eval(input())
    result = solve(l)
    print(result)
