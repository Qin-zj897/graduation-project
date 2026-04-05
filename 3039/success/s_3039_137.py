def solve(ls):
    n,m=max(ls),min(ls)
    while n in ls:
        ls.remove(n)
    while m in ls:
        ls.remove(m)
    return ls


if __name__ == '__main__':
    ls = eval(input())
    result = solve(ls)
    print(result)
