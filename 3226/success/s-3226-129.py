def search(nums):
      for i in nums:
           b=nums.count(i)
           n=[]
           n.append(b)
      if max(n)>len(nums)/2:
           return(i)
      else: 
            return("False")





nums = eval(input())
y = search(nums)
print(y)


