def search(nums:list):
    r=False
    l=len(nums)
    for x in nums:
        c=nums.count(x)
        if c>l/2:
            r=x
    return r





nums = eval(input())
y = search(nums)
print(y)


