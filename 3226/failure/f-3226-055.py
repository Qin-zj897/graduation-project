def search(nums):
    a=len(nums)//2
    for x in nums:
        c=nums.count(x)
        if c>a:
            return x





nums = eval(input())
y = search(nums)
print(y)


