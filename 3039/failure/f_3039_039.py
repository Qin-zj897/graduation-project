def solve(list0):
    list0=list(list0)
    a=max(list0)
    b=min(list0)
    for i in list0:
        if i==a :
            list0.pop(i)
    for i in list0:
        if i==b :
            list0.pop(i)
    return list0


if __name__ == '__main__':
    list0 = eval(input())
    result = solve(list0)
    print(result)
