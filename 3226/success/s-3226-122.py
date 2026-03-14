def search(nums):
    n1=[]
    for x in nums:
        n1.append(nums.count(x))
        n2=max(n1)    
    if n2>len(nums)/2:
       return max(nums,key=nums.count)  
    else:
        return  "False"





nums = eval(input())
y = search(nums)
print(y)


