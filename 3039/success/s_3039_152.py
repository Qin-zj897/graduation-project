def solve(lst):
    lst1=[]
    x=min(lst)
    y=max(lst)
    for i in lst:
        if i!=x and i!=y:
            lst1.append(i)
    return lst1


if __name__ == '__main__':
    lst = eval(input())
    result = solve(lst)
    print(result)
