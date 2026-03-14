def search(nums):
       n=len(nums)
       for x in nums:
             if nums.count(x) > n//2:
                  h=x
             else:
                  h="False"
       return h





nums = eval(input())
y = search(nums)
print(y)


