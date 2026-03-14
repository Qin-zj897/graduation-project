def search(n):
    num=0
    an=0
    for x in n:
        if n.count(x)>len(n)/2:
            num+=1
            an=x
    if num==0:
        an='False'
    return an





nums = eval(input())
y = search(nums)
print(y)


