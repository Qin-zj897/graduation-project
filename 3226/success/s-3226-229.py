def search(nums):
       n=len(nums)
       for i in nums:
             if nums.count(i)>n//2:
                 a=i
             else:
                  a="False"
             return a





nums = eval(input())
y = search(nums)
print(y)


