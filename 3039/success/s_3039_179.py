def solve(list1):
    a=max(list1)
    b=min(list1)
    list2=list1.copy()
    for i in list1:
        if i==a or i==b:
            list2.remove(i)
    return list2


if __name__ == '__main__':
    list1 = eval(input())
    result = solve(list1)
    print(result)
