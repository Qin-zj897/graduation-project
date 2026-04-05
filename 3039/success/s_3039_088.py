def solve(a):
    b=a.copy()
    max=max(b)
    min=min(b)
    for x in b:
        if max in a:
            a.remove(max)
    for x in b:
        if min in a:
            a.remove(min)
    return a


if __name__ == '__main__':
    a = eval(input())
    result = solve(a)
    print(result)
