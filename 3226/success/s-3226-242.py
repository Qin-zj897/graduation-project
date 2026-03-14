def search(nums):
    for x in nums:
        if nums.count(x)>len(nums)/2:
            h=x
            return h
        else:
            h=False
    return h    





nums = eval(input())
y = search(nums)
print(y)


