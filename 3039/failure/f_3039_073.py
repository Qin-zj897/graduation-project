def solve(ls):
    a=max(ls)
    b=min(ls)
    c=ls.count(a)
    d=ls.count(b)
    for i in range(c):
        ls.remove(a)
        ls1=ls
    for i in range(d):
        ls1.remove(b)
        ls2=ls1
    return ls2


if __name__ == '__main__':
    ls = eval(input())
    result = solve(ls)
    print(result)
