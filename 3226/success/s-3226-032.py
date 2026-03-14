def search(nums):
    for x in nums:
        if int(nums.count(x))>float(len(nums)/2):
            return x
        else:
            return False





nums = eval(input())
y = search(nums)
print(y)


