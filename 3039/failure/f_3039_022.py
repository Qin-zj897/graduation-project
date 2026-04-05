def solve(ls):
    ls.sort()
    m=ls.count(ls[0])
    n=ls.count(ls[-1])
    for i in range(m):
        del ls[0]
    for i in range(n):    
        del ls[-1]
    return ls


if __name__ == '__main__':
    ls = eval(input())
    result = solve(ls)
    print(result)
