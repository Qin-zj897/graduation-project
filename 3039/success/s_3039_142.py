def solve(ls2):
    ls1=ls2.copy()
    for i in ls1:
        if i>=max(ls1)or i<=min(ls1):
            ls2.remove(i)
    return ls2


if __name__ == '__main__':
    ls2 = eval(input())
    result = solve(ls2)
    print(result)
