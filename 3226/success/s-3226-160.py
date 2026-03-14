def search(nums):
    ls=[]
    a=len(nums)
    for i in nums:
        if nums.count(i)>a//2:
           ls.append(i)
    if len(ls)<1:
        y=False
    else:
        y=ls[0]
    return y





nums = eval(input())
y = search(nums)
print(y)


