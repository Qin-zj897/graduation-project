def search(nums):
    for num in nums:
        if nums.count(num)>len(nums)//2:
            return num
        else:
            return False





nums = eval(input())
y = search(nums)
print(y)


