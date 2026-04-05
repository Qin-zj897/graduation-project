def solve(ls):
    maxnum=max(ls)
    minnum=min(ls)
    a=[maxnum,minnum]
    ls1=[]
    for i in ls:
        if i not in a:
            ls1.append(i)
    return ls1


if __name__ == '__main__':
    ls = eval(input())
    result = solve(ls)
    print(result)
