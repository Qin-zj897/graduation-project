def solve(ls1):
    a = max(ls1)
    b = min(ls1)
    ls2 = []
    for i in ls1:
        if i not in ls2:
            ls2.append(i)
    for i in ls2:
        if i == a:
            del(i)
    else:
        pass
    for i in ls2:
        if i == b:
            del(i)
    else:
        pass
    return ls1


if __name__ == '__main__':
    ls1 = eval(input())
    result = solve(ls1)
    print(result)
