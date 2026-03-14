def search(nums):
    new = []
    w = len(nums) / 2
    for x in nums:
        a = nums.count(x)
        if a > w:
            new.append(x)
    if new == []:
        return False
    else:
        return new[0]





nums = eval(input())
y = search(nums)
print(y)


