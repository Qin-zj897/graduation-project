def solve(ls):
    ma = max(ls)
    mi = min(ls)
    ls1 = ls.copy()
    for num in ls1:
        if num == ma or num == mi:
            ls.remove(num)
    return ls


if __name__ == '__main__':
    ls = eval(input())
    result = solve(ls)
    print(result)
