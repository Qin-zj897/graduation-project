def search(nums):
    n=len(nums)
    
    for i in range(n):
        if nums.count(nums[i])>(n//2):
            a=nums[i]
            return(a)
    return("False")    





nums = eval(input())
y = search(nums)
print(y)


