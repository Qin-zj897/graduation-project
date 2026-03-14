def search(nums):
    for i in range(len(nums)):
        a=nums.count(nums[i])
        if a>(len(nums)//2):
            return nums[i]
            break
        else:
            return False





nums = eval(input())
y = search(nums)
print(y)


