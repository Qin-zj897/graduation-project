def solve(ls):
    b=max[ls]
    a=ls.count(b)
    for i in range(a):
        ls.remove(b)
    c=min[ls]
    d=ls.count(c)
    for i in range(d):
        ls.remove(c)
    return ls


if __name__ == '__main__':
    ls = eval(input())
    result = solve(ls)
    print(result)
