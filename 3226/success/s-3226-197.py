def search(nums):
      n = len(nums)
      b = n//2
      for i in nums:
            if nums.count(i) > b:
                c=i
            else:
                c="False"
      return c





nums = eval(input())
y = search(nums)
print(y)


