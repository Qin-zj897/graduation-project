def search(nums):
    for i in nums:
        a = nums.count(i)
        if a>=1/2*len(nums):
            return(i)
    else:
        return('False')





nums = eval(input())
y = search(nums)
print(y)


