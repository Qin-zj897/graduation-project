def search(nums):
       n = int(len(nums)/2)
       for i in nums:
             if nums.count(i)>n:
                  h = i
             else:
                  h = "False"
       return  h





nums = eval(input())
y = search(nums)
print(y)


