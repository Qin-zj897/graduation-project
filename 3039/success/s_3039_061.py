def solve(a):
    c = []
    for x in a:
        if x != max(a) and x != min(a):
            c.append(x)
    return c


if __name__ == '__main__':
    a = eval(input())
    result = solve(a)
    print(result)
