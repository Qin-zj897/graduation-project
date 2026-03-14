def search(nums):
    lst1 = []
    icount = 0
    for x in nums:
        if nums.count(x)>len(nums)//2:
            lst1.append(x)
            break
    if len(lst1) == 0:
        return False
    else:
       return lst1[0]





nums = eval(input())
y = search(nums)
print(y)


