def search(nums):
    lst=[]
    for x in nums:
        if nums.count(x) > len(nums)//2:
           lst.append(x)
    if lst==[]:
        return "False"
    else:
        return x





nums = eval(input())
y = search(nums)
print(y)


