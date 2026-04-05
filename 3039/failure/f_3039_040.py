def solve(list0):
    from copy import deepcopy
    list0=list(list0)
    a=max(list0)
    b=min(list0)
    list1=deepcopy(list0)
    for i in list1:
        if i==a :
            list0.remove(i)
    for i in list1:
        if i==b :
            list0.remove(i)
    return list0


if __name__ == '__main__':
    list0 = eval(input())
    result = solve(list0)
    print(result)
