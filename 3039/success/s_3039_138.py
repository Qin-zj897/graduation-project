def solve(lst):
    a=max(lst)
    b=min(lst) 
    lst1=lst.copy()
    for i in lst:
        if i==a or i==b:
            lst1.remove(i)
    return lst1


if __name__ == '__main__':
    lst = eval(input())
    result = solve(lst)
    print(result)
