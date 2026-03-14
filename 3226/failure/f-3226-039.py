def search(nums):
    ls = []
    for x in nums:
        if nums.count(x) > 1:
            a = x
            ls.append(x)
        else:
            pass
    if ls!=[] :
        pass
    else:
        a = "False"
    return(a)





nums = eval(input())
y = search(nums)
print(y)


