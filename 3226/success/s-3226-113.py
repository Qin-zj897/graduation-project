def search(n):
    b=[]
    for x in n:
        if n.count(x)>len(n)/2:
            if x not in b:
                b.append(x)
    if len(b)>0:
        c=b[0]
    else:
        c=False
    return c





nums = eval(input())
y = search(nums)
print(y)


