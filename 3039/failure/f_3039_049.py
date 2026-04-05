def solve(a):
    '''a=eval(read())
    b=a.count(max(a))
    c=a.count(min(a))
    if len(a)>0:
        for x in range(b):
            a.remove(max(a))
    if len(a)>0:
        for x in range(c):
            a.remove(min(a))
    write(a)'''

    for x in a:
        if x==max(a) or x==min(a):
            for i in range(a.count(x)):
                a.remove(x)
    return a


if __name__ == '__main__':
    a = eval(input())
    result = solve(a)
    print(result)
