def solve(ls1):
    ls2=[]
    a=max(ls1)
    b=min(ls1)
    for i in ls1:
        if i!=a and i!=b:
            ls2.append(i)
    return ls2


if __name__ == '__main__':
    ls1 = eval(input())
    result = solve(ls1)
    print(result)
