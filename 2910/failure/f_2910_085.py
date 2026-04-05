def solve(h, n):
    '''s=list(read())
    b=[]
    for x in s:
        a=str((int(x)+5)%10)
        b.append(a)
    b.reverse()
    write(''.join(b))'''
    c=[]
    for x in range(1,n+1):
        a=h*0.5**x
        b=2*a
        c.append(b)
    d=sum(c)
    f=h+d
    return '%.2f'%(f)


if __name__ == '__main__':
    h = eval(input())
    n = eval(input())
    result = solve(h, n)
    print(result)
