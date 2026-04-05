def solve(lst):
    a=max(lst)
    b=min(lst)
    res=[]
    for i in lst:
        if i!=a and i!=b:
            res.append(i)
    return res


if __name__ == '__main__':
    lst = eval(input())
    result = solve(lst)
    print(result)
