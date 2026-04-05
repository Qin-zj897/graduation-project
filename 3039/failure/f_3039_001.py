def solve(ls1):
    a=max(ls1)
    b=min(ls1)
    for i in ls1:
        if i==a or i==b:
            ls1.remove(i)
        else:
            pass
    return ls1


if __name__ == '__main__':
    ls1 = eval(input())
    result = solve(ls1)
    print(result)
