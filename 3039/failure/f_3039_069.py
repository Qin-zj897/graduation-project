def solve(lst):
    G = max(lst)
    g = min(lst)
    maxnum = lst.count(max(lst,key = lst.count))
    minmum = lst.count(min(lst,key = lst.count))
    for i in range(maxnum):
        lst.remove(G)
    for i in range(minmum):
        lst.remve(g)
    return lst


if __name__ == '__main__':
    lst = eval(input())
    result = solve(lst)
    print(result)
