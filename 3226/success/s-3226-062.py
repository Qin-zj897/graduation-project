def search(nums):
    n=len(nums)
    for x in nums:
        if nums.count(x)>(n//2):
            m=x
        else:
             m=False
    return m





nums = eval(input())
y = search(nums)
print(y)


