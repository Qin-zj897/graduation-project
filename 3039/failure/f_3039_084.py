def solve(lst):
    lst=read().split()
    a=max(lst)
    b=min(lst)
    list1=lst.copy
    for x in lst:
        if x==a or x==b:
            list1.remove(x)
        return list1


if __name__ == '__main__':
    lst = eval(input())
    result = solve(lst)
    print(result)
