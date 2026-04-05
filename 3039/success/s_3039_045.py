def solve(list1):
    m=max(list1)
    n=min(list1)
    list2=list1.copy()
    for x in list1:
        if x==m or x==n:
            list2.remove(x)
    return list2


if __name__ == '__main__':
    list1 = eval(input())
    result = solve(list1)
    print(result)
