def solve(a):
    b = max(a)
    c = min(a)
    i = 0
    while i < len(a):
        if b == a[i]:
            a.remove(b)
            i = i - 1
        elif c == a[i]:
            a.remove(c)
            i = i - 1
        i = i + 1
    return a


if __name__ == '__main__':
    a = eval(input())
    result = solve(a)
    print(result)
