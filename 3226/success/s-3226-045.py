def search(nums):
    count = 0
    candidate =None
    for x in nums:
        if count == 0:
            candidate = x
        if count ==x:
            count +=1
        else:
            count -= 1
    a = nums.count(candidate)
    if a>len(nums)/2:
        return candidate
    else:
        return False





nums = eval(input())
y = search(nums)
print(y)


