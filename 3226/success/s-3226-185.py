def search(nums):
    ls=[]
    for i in nums:
        if nums.count(i)>len(nums)//2:
            ls.append(i)
            return i 
    if len(ls)==0:
            return False





nums = eval(input())
y = search(nums)
print(y)


