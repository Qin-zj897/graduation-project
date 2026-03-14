def search(nums):
    for i in nums:
        if nums.count(i)>len(nums)*0.5:
            return i
    return False





nums = eval(input())
y = search(nums)
print(y)


