def solve(lst):
    a = max(lst)
    b = min(lst)
    lst1 = []
    for x in lst:
        if x!=a and x!=b:
            lst1.append(x)
        else:
            pass
    return lst1


if __name__ == '__main__':
    lst = eval(input())
    result = solve(lst)
    print(result)
