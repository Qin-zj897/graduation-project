def search(nums):
    for i in nums:
        if nums.count(i)>(len(nums)/2):
            h=i
        else:
            h=False
    return h





nums = eval(input())
y = search(nums)
print(y)


