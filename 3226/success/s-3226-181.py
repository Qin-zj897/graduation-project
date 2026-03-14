def search(nums):
    a=len(nums)//2
    for i in nums:
      b=nums.count(i)
    if b>a:
      return(i)
    else:
      return(False)






nums = eval(input())
y = search(nums)
print(y)


