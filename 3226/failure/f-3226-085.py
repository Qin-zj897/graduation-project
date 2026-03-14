def search(nums):
    n=len(nums)
    for i in nums:
        if nums.counts(i)>n//2:
            i=y
        else:
            y=False
        return y





nums = eval(input())
y = search(nums)
print(y)


