def solve(ls):
    m = max(ls)
    n = min(ls)
    while m in ls:
        ls.remove(m)
    while n in ls:
        ls.remove(n) 
    return ls


if __name__ == '__main__':
    ls = eval(input())
    result = solve(ls)
    print(result)
