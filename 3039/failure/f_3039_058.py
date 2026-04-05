def solve(ls1):
    ls2 = []
    for i in ls1:
        if i not in ls2:
            ls2.append(i)
    else:
        pass
    a = max(ls2)
    b = min(ls2)
    ls2.remove(a)
    if len(ls2)<1:
        return "[]"
    else:
        ls2.remove(b)
        return ls2


if __name__ == '__main__':
    ls1 = eval(input())
    result = solve(ls1)
    print(result)
