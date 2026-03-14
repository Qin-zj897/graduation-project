def search(nums):
    a=len(nums)//2
    for x in nums:
        c=nums.count(x)
        if c>a:
            m=x
        else:
            m=False
            return m





nums = eval(input())
y = search(nums)
print(y)


