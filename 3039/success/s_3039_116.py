def solve(ls1):
    ls2=[]
    a=min(ls1)
    b=max(ls1)
    #write(a,b)
    for i in ls1:
        if i!=a and i!=b : ls2.append(i)
    return ls2


if __name__ == '__main__':
    ls1 = eval(input())
    result = solve(ls1)
    print(result)
