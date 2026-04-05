def solve(a):
    a.sort()
    b = a[0]
    c = a[-1]
    if b in a:
        a.remove(b)
    if c in a:
        a.remove(c)
    elif b not in a and c not in a:
        return a


if __name__ == '__main__':
    a = eval(input())
    result = solve(a)
    print(result)
