def search(nums):
   for i in nums:
      if nums.count(i)<=len(nums)/2:
         return False
      else:
         return i
 





nums = eval(input())
y = search(nums)
print(y)


