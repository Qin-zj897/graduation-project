def solve(ls):
    a=max(ls)
    b=min(ls)
    for a in ls:
        ls.remove(a)
    for b in ls:
        ls.remove(b)
    return ls


if __name__ == '__main__':
    ls = eval(input())
    result = solve(ls)
    print(result)
