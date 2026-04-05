def solve(lst):
    a=read()
    a.remove(max(a))
    a.remove(min(a))
    return a


if __name__ == '__main__':
    lst = eval(input())
    result = solve(lst)
    print(result)
