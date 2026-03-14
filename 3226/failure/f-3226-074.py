def search(nums):
    n=len(nums)
    for i in nums:
        if nums.count(i)>n//2:
            y=i
        else:
            y="False"
    return y
nums  =  eval(input())
y  =  search(nums)
print(y)





nums = eval(input())
y = search(nums)
print(y)


