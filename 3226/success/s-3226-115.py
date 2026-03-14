def search(nums):
    for x in nums:
        s=0
        if nums.count(x) > len(nums) // 2:
            d=x
            s += 1
        else:
            pass
    if s == 0:
        d="False"
    return d






nums = eval(input())
y = search(nums)
print(y)


