def search(nums):
    for i in nums:
        c=nums.count(i)
        if c>0.5*len(nums):
            return i
        else:
            return "False"





nums = eval(input())
y = search(nums)
print(y)


