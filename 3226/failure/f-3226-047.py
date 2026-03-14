def search(nums):
    for x in nums:
        if nums.count(x) > 1:
            a = x
    if a==0 :
        pass
    else:
        a = "False"
    return(a)





nums = eval(input())
y = search(nums)
print(y)


