def solve(a):
    b =[]
    for x in a:
        if x != max(a) and x!=min(a):
            b.append(x)
    return b


if __name__ == '__main__':
    a = eval(input())
    result = solve(a)
    print(result)
