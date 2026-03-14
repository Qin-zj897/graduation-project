def search(nums):
    for x in nums:
        c=nums.count(x)
        if c>0.5*len(nums):
            return x
        else:
            return "False"






nums = eval(input())
y = search(nums)
print(y)


