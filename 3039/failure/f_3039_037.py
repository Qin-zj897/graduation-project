def solve(lst):
    a = max(lst)
    b = min(lst)
    l2 = []
    for i in lst:
        if i == a:
            continue
        else:
            l2.append(i)
    for i in lst:
        if i == b:
            continue
        else:
            l2.append(i)        
    return l2


if __name__ == '__main__':
    lst = eval(input())
    result = solve(lst)
    print(result)
