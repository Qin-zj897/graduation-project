def solve(lst):
    a=max(lst)
    b=min(lst)
    lst1=lst.copy()
    for x in lst:
        if x==a or x==b:
           lst1.remove(x)
    return lst1


if __name__ == '__main__':
    lst = eval(input())
    result = solve(lst)
    print(result)
