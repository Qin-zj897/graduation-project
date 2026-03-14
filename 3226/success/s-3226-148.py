def search(nums):
    b=[]
    for i in nums:
        b.append(nums.count(i))
        if max(b)>len(nums)//2:
            return(i)
        else:
            return('False')





nums = eval(input())
y = search(nums)
print(y)


