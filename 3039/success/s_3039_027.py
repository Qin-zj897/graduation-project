def solve(a):
    m=max(a)
    n=min(a)
    while True:
        if m in a:
            a.remove(m)
        else:
            break
    while True:
        if n in a:
            a.remove(n)
        else:
            break
    return a


if __name__ == '__main__':
    a = eval(input())
    result = solve(a)
    print(result)
