def search(x):
    a=len(x)
    b=0
    for i in x:
        if x.count(i)>a//2:
            b=i
    if b!=0:
        return b
    elif b==0:
        return "False"





nums = eval(input())
y = search(nums)
print(y)


