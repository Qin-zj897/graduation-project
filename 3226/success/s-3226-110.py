def search(nums):
    x=0
    b=[]
    for i in range (len(nums)):
        a=nums.count(nums[i])
        if a >len(nums)/2:
            x+=1
            b.append(nums[i])    
        else:
            pass
    if not x==0:
        return b[0]     
    else: 
        return False





nums = eval(input())
y = search(nums)
print(y)


