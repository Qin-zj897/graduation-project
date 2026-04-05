def solve(lst):
    a=max(lst)
    b=min(lst)
    while a in lst :
        lst.remove(a)
    while b in lst :
        lst.remove(b)
    return lst


if __name__ == '__main__':
    lst = eval(input())
    result = solve(lst)
    print(result)
