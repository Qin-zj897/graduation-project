def search(a):
    b=max(a,key=a.count)
    f=a.count(b)
    d=len(a)
    e=d//2
    if f>e:
        return b
    else :
        return "False"





nums = eval(input())
y = search(nums)
print(y)


