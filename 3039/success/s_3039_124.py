def solve(ls):
    ls1=[]
    a=max(ls)
    b=min(ls)
    for x in ls:
        if x!=a and x!=b:
            ls1.append(x)
    return ls1


if __name__ == '__main__':
    ls = eval(input())
    result = solve(ls)
    print(result)
