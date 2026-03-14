def search(nums):
    a=0
    le=len(nums)//2
    for x in nums:
        if nums.count(x)>le:
            a=x
        elif a!=0:
            return "False"
            





nums = eval(input())
y = search(nums)
print(y)


