def search(nums):
    a=0
    for x in nums:
        if nums.count(x)>a:
            a=nums.count(x)
            c=x
    b=len(nums)
    if a>b//2:
            h=c
    else:
            h=False
    return h





nums = eval(input())
y = search(nums)
print(y)


