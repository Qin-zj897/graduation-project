def search(a):
    n=len(a)
    ls1=[]
    for x in a:
        c=a.count(x)
        ls1.append(c)
    d=max(ls1)
    if d>n/2:
        for x in a:
            if a.count(x)==d:
                return x
    else:
        return "False"





nums = eval(input())
y = search(nums)
print(y)


