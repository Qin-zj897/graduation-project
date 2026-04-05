def solve(list):
    a=max(list)
    b=min(list)
    list1=list.copy()
    for x in list:
        if x==a or x==b:
            list1.remove(x)
    return list1


if __name__ == '__main__':
    list = eval(input())
    result = solve(list)
    print(result)
