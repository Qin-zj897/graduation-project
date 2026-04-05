def solve(h, n):
    d=h
    if n>1:
        for x in range(1,n):
            c=d*(1/2)**(x)
            d=d+c*2
        return '%.2f'%(d)
    else:
        return '%.2f'%(d)


if __name__ == '__main__':
    h = eval(input())
    n = eval(input())
    result = solve(h, n)
    print(result)
