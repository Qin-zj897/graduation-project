def search(nums):
      n=len(nums)
      for i in nums:
           c=sum(1 for element in nums if element==i)
           if c>n/2:
              return(i)
           else:
              return('False')





nums = eval(input())
y = search(nums)
print(y)


