def search(nums):
    n=len(nums)
    for x in nums:
        if nums.count(x)>(n/2):
            y=x
        else:
            y=False
        return y





nums = eval(input())
y = search(nums)
print(y)


