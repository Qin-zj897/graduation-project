def search(nums):
    c=0
    for i in nums:
        if nums.count(i)>len(nums)//2:
            c=1
            return i
            break
    if c==0:
        return "False"





nums = eval(input())
y = search(nums)
print(y)


