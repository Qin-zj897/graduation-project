def solve(lst):
    m = max(lst)
    n = min(lst)
    for i in range(lst.count(m)):
        lst.remove(m)
    for i in range(lst.count(n)):
        lst.remove(n)
    return lst


if __name__ == '__main__':
    lst = eval(input())
    result = solve(lst)
    print(result)
