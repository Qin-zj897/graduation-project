def solve(h, m):
    s=0
    x=2*h
    for i in range(m):
        s=s+x
        x=x*0.5
    return '$.2f'%(s-h)


if __name__ == '__main__':
    h = eval(input())
    m = eval(input())
    result = solve(h, m)
    print(result)
