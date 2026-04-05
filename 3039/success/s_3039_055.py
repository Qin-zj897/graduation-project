def solve(lst):
    G = max(lst)
    g = min(lst)
    maxnum = lst.count(G)
    minmum = lst.count(g)
    if G>g:
        for i in range(maxnum):
            lst.remove(G)
        for i in range(minmum):
            lst.remove(g)
    else:
        for i in range(maxnum):
            lst.remove(G)
    return lst


if __name__ == '__main__':
    lst = eval(input())
    result = solve(lst)
    print(result)
