def search(nums):
    for i in nums:
        a = int(len(nums))
        if nums.count(i) > a//2 :
            return i
        else:
            return False





nums = eval(input())
y = search(nums)
print(y)


