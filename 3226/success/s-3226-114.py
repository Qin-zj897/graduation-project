def search(n):
    b=[]
    for x in n:
        if n.count(x)>len(n)/2:
            c=x
        else:
            c=False
    return c





nums = eval(input())
y = search(nums)
print(y)


