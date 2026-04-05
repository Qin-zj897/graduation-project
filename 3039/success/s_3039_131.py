def solve(b):
    '''a=read().split(',')
    a=list(a)
    m=[]
    for x in range(a):
        n=[]
        n.append(a[x])
        n.append(b[x])
        m.append(n)
    write(m)'''

    '''n,m,l=map(int,read().split(','))
    c=[]
    for i in range (m):
        d=n+l*i
        c.append(d)
    write(c)'''

    '''a=eval(read())
    n,m=map(int,read().split(','))
    if n<=len(a):
        for x in range (n,m):
            b=a.pop(x)
        return a
    else:
        write('error')'''

    '''a=eval(read())
    b=[]
    for i in a:
        if i >=2:
            for j in range(2,i,1):
                if i%j==0:
                    break
            else:
                b.append(i)
    write(b)'''

    '''a=eval(read())
    b=sum(a)
    c=len(a)
    d=b/c
    if b%c==0:
        return '%d'%(b/c)
    else:
        write('%.2f'%(d)) '''

    '''a=list(map(str,read().split()))
    n,m=map(int,read().split())
    a[n],a[m]=a[m],a[n]
    write(a)'''

    a=eval(read())
    b=a.count(max(a))
    c=a.count(min(a))
    if len(a)>0:
        for x in range(b):
            a.remove(max(a))
    if len(a)>0:
        for x in range(c):
            a.remove(min(a))
    return a


if __name__ == '__main__':
    b = eval(input())
    result = solve(b)
    print(result)
