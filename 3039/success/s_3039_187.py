def solve(lst):
    # lst = eval(read())
    # m = max(lst)
    # n = min(lst)
    # for i in range(lst.count(m)):
    #     lst.remove(m)
    # for i in range(lst.count(n)):
    #     lst.remove(n)
    # write(lst)

    lst2 = []
    m = max(lst)
    n = min(lst)
    for i in range(len(lst)):
        if lst[i] != m and lst[i] != n:
            lst2.append(lst[i])
    return lst2


if __name__ == '__main__':
    lst = eval(input())
    result = solve(lst)
    print(result)
