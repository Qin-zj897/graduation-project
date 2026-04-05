def solve(lst):
    lst=read().split()
    a=max(lst)
    b=min(lst)
    for x in lst:
        if x==a or x==b:
            lst.remove(x)
        return lst


if __name__ == '__main__':
    lst = eval(input())
    result = solve(lst)
    print(result)
