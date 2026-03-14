def search(nums):
   count = 0
   candidate = None
   for num in nums:
      if count == 0:
         num = candidate
      if num ==candidate:
         count +=1
      else:
         count -=1
   count = nums.count(candidate)   
   if count > len(nums)//2:
     return candidate
   else:
     return False
     





nums = eval(input())
y = search(nums)
print(y)


