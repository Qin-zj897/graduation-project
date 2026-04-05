def solve(lst1):
    a = max(lst1)
    b = min(lst1)
    for i in range(len(lst1)):
        if a in lst1:
            lst1.remove(a)
        else:
            break
    for j in range(len(lst1)):
        if b in lst1:
            lst1.remove(b)
        else:
            break
    return lst1


if __name__ == '__main__':
    lst1 = eval(input())
    result = solve(lst1)
    print(result)
