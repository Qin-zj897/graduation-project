def search(nums):
    a=len(nums)//2
    for x in nums:
        c=nums.count(x)
        if c>a:
            return x
        else:
            print("False")





nums = eval(input())
y = search(nums)
print(y)


