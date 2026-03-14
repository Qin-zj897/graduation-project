def search(n):
    list=[]
    for x in n:
        b=n.count(x)
        list.append(b)
    c=max(list)    
    if c>(len(n)/2):
        return c
    else:
        return False 





nums = eval(input())
y = search(nums)
print(y)


