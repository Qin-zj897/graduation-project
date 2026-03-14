def search(nums):
    n=len(nums)
    b=[]
    for a in nums:
        if nums.count(a)>n//2:
            b.append(a)
        else:
            pass
    if b==[]:
        c="False"
    else:
        c=max(b)
    return c





nums = eval(input())
y = search(nums)
print(y)


