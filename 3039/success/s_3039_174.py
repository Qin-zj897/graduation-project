def solve(lst):
    x1=max(lst)
    x2=min(lst)
    for i in range(lst.count(x1)):
        lst.remove(x1)
    for x in range(lst.count(x2)):
        lst.remove(x2)
    return lst


if __name__ == '__main__':
    lst = eval(input())
    result = solve(lst)
    print(result)
