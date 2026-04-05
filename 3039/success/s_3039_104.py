def solve(lst):
    a=max(lst)
    b=min(lst)
    lst2=lst.copy()
    for i in lst:
        if i==a or i==b:
            lst2.remove(i)
    return lst2


if __name__ == '__main__':
    lst = eval(input())
    result = solve(lst)
    print(result)
