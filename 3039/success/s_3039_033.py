def solve(a):
    b = []
    for i in a:
        if i <max(a) and i>min(a):
            b.append(i)
    return b


if __name__ == '__main__':
    a = eval(input())
    result = solve(a)
    print(result)
