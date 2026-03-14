def search(nums):
    n = len(nums)
    for i in nums:
        if nums.count(i)>(n/2):
            m = i
        else:
            m = False
    return m






nums = eval(input())
y = search(nums)
print(y)


