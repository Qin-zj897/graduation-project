def search(nums):
    n=len(nums)
    for i in nums:
        if nums.count(i)>n//2:
            h=i
        else:
            h="False"
    return h
nums  =  eval(input())
y  =  search(nums)
print(y)





nums = eval(input())
y = search(nums)
print(y)


