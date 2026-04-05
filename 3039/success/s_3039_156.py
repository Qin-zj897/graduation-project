def solve(ls1):
    m = max(ls1)
    n = min(ls1)
    ls2=ls1.copy()
    for x in ls1:
        if x == m or x == n:
            ls2.remove(x)
    return ls2


if __name__ == '__main__':
    ls1 = eval(input())
    result = solve(ls1)
    print(result)
