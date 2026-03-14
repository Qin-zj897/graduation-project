def search(nums):
    for i in nums:
        if nums.count(i)<len(nums)//2:
             continue
        elif nums.count(i)>len(nums)//2:
             return i
    return False
 





nums = eval(input())
y = search(nums)
print(y)


