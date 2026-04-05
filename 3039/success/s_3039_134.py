def solve(a):
    el = min(a)
    e2 = max(a)
    b = a.count(el)
    for i in range(b):
        a.remove(el)
    b = a.count(e2)
    for i in range(b):
        a.remove(e2)
    return a


if __name__ == '__main__':
    a = eval(input())
    result = solve(a)
    print(result)
