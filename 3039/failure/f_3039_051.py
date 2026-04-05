def solve(list1):
    m=max(list1)
    n=min(list1)
    for x in list1:
        if x==m or x==n:
            list1.remove(x)
    return list1


if __name__ == '__main__':
    list1 = eval(input())
    result = solve(list1)
    print(result)
