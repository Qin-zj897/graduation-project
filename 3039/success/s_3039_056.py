def solve(ls):
    ma=max(ls)
    mi=min(ls)
    num=0
    for i in range(len(ls)):
        for x in ls:
            if x==ma or x==mi:
                ls.remove(x)
    return ls


if __name__ == '__main__':
    ls = eval(input())
    result = solve(ls)
    print(result)
