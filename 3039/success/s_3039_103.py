def solve(lst):
    lst2=[]
    a=max(lst)
    b=min(lst)
    for i in lst:
        if i!=a and i!=b:
            lst2.append(i)
    return lst2


if __name__ == '__main__':
    lst = eval(input())
    result = solve(lst)
    print(result)
