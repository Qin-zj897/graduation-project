def solve(ls1):
    n=max(ls1)
    m=min(ls1)
    ls2=[]
    for i in ls1:
        if i!=n and i!=m:
            ls2.append(i)
    return ls2


if __name__ == '__main__':
    ls1 = eval(input())
    result = solve(ls1)
    print(result)
