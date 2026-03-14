def search(nums):
    a=list(nums)
    c=[]
    n=len(a)
    for x in a:
        b=a.count(x)
        c.append(b)
    e=max(c)
    if e > n/2:
        d=1
        g=1
        for x in a:
            b=a.count(x)
            if b>=d:
                g=x
            return g
    else: return "False"





nums = eval(input())
y = search(nums)
print(y)


