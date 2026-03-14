def search(nums):
   count = 0
   candidate = None
   for num in nums:
      if count == 0:
         candidate = num
      if candidate == num:
         count += 1
      else:
         count-=1
  count = candidate.count()
   if count>len(nums)//2
     return   candidate
   else:
     return False






nums = eval(input())
y = search(nums)
print(y)


