def search(nums):
    n=len(nums)
    ls=[]
    ls2=[]
    for i in nums:
        a=nums.count(i)
        ls.append(a)
    b=max(ls)
    if b>n//2:
        for j in nums:
            if j not in ls2:
               ls2.append(j) 
            c=nums.count(j)
            if c>n/2:
                return j
    else:
        return "False"   





nums = eval(input())
y = search(nums)
print(y)


