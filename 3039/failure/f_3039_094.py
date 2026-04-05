def solve(lst):
    t=read()
    t=t.replace("[","").replace("]","")
    t=t.split(',')
    t=list(map(int,t))
    a=max(t)
    b=min(t)
    p=[]
    for i in t:
        if i not in p:
            p.append(i)
    for x in t:
        if x!=a:
            p.remove(x)
        else:
            pass
    for y in p:
        if y==b:
            p.remove(y)
    return p


if __name__ == '__main__':
    lst = eval(input())
    result = solve(lst)
    print(result)
