def search(nums):
    a=len(nums)//2
    b=0
    c=len(nums)
    for x in nums:
        b+=1
        if nums.count(x) > a:
            return x
        elif b == c:
            return "False"





nums = eval(input())
y = search(nums)
print(y)


