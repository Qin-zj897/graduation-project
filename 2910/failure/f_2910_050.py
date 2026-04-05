def solve(h, n):
    if n>1:
        for x in range(1,n):
            c=h*(1/2)**(x)
            h=h+c*2
        return '%.2f'%(h)
    else:
        return '%.2f'%(h)


if __name__ == '__main__':
    h = eval(input())
    n = eval(input())
    result = solve(h, n)
    print(result)
