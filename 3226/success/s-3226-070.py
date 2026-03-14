def search(a):
    x=0
    for i in a:
        if a.count(i)>len(a)//2:
             x=i
        else:
             x="False"
    return x





nums = eval(input())
y = search(nums)
print(y)


