def solve(ls1):
    a=max(ls1)
    b=min(ls1)
    ls2=ls1.copy()
    for x in ls1:
        if x==a or x==b:
            ls2.remove(x)
    return ls2


if __name__ == '__main__':
    ls1 = eval(input())
    result = solve(ls1)
    print(result)
