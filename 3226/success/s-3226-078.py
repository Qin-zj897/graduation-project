def search(nums):
    for x in nums:
        a=nums.count(x)
        if a>len(nums)/2:
            return x
            break
    else:
        return False





nums = eval(input())
y = search(nums)
print(y)


