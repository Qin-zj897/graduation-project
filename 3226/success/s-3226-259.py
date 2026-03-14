def search(nums):
      for i in nums:
            if nums.count(i)>len(nums)//2:
               return i
            else:
               return False
           
      return t





nums = eval(input())
y = search(nums)
print(y)


