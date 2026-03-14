def search(nums):
    n=len(nums)
    a=n//2
    b=0
    for i in nums:
        b=nums.count(i)
        if a<b:
          b=i
        else:
           b="False"
    return(b)





nums = eval(input())
y = search(nums)
print(y)


