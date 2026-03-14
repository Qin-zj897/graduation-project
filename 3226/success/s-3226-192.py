def search(nums):
    lit=[]
    for i in nums:
        a=nums.count(i)
        if a>(len(nums)//2):
            lit.append(str(i))
    if lit==[]:
        return False
    else:
        return(lit[0])





nums = eval(input())
y = search(nums)
print(y)


