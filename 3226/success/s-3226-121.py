def search(nums):
    for x in nums:
        b=nums.count(x)
        n=[]
        n.append(b)
    if max(n)>len(nums)/2:
        return (x)
    else:
        return("False")
        





nums = eval(input())
y = search(nums)
print(y)


