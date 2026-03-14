def search(nums):
    for x in nums:
        count = nums.count(x)
        if count>len(nums)//2:
            return(x)
        else:
            return(False)





nums = eval(input())
y = search(nums)
print(y)


