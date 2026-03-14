def search(ls):
    a=len(ls)//2
    kong=[]
    for x in ls:
        b=ls.count(x)
    if  b>a:
        kong.append(x)
        return x
    if b<=a:
        return False





nums = eval(input())
y = search(nums)
print(y)


