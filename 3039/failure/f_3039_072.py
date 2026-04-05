def solve(lst):
    G = max(lst)
    g = min(lst)
    maxnum = lst.count(G)
    minmum = lst.count(g)
    for i in range(maxnum):
        lst.remove(G)
    for i in range(minmum):
        lst.remove(g)
    return lst


if __name__ == '__main__':
    lst = eval(input())
    result = solve(lst)
    print(result)
