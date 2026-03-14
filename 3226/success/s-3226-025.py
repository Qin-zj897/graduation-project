def search(nums):
    ls = []
    n = len(nums)
    for x in nums:
        if nums.count(x) > n/2:
            a = x
            ls.append(x)
        else:
            pass
    if ls==[] :
        a = "False"
    else:
        pass
    return(a)





nums = eval(input())
y = search(nums)
print(y)


