def search(nums):
    for i in nums:
        a = int(nums.count(i))
        b = int(len(nums))//2
        if a>b:
            return i
        else:
            return False
        





nums = eval(input())
y = search(nums)
print(y)


