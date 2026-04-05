def solve(lst):
    a=max(lst)
    b=min(lst)
    for i in lst[:]:
        if i==a or i==b:
            lst.remove(i)
    return lst


if __name__ == '__main__':
    lst = eval(input())
    result = solve(lst)
    print(result)
