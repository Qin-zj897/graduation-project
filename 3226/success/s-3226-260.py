def search(nums):
    for x in nums:
        a=nums.count(x)
        n=len(nums)
        if a>n//2:
            return x
        if a<=n//2:
            return False
      





nums = eval(input())
y = search(nums)
print(y)


