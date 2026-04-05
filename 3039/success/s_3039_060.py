def solve(ls):
    a = max(ls)
    b = min(ls)
    ls2 = []
    for x in ls:
        if x != a and x != b:
            ls2.append(x)
    return ls2


if __name__ == '__main__':
    ls = eval(input())
    result = solve(ls)
    print(result)
