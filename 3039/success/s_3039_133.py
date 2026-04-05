def solve(lst):
    lst1=lst.copy()
    maxl=max(lst)
    minl=min(lst)
    for i in lst1:
        if i==maxl or i==minl:
            lst.remove(i)
    return lst


if __name__ == '__main__':
    lst = eval(input())
    result = solve(lst)
    print(result)
