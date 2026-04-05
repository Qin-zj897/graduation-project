def solve(ls1):
    a=max(ls1)
    b=min(ls1)
    ls2=ls1.copy()
    for i in ls2:
        if i==a or i==b:
            ls1.remove(i)
    return ls1


if __name__ == '__main__':
    ls1 = eval(input())
    result = solve(ls1)
    print(result)
