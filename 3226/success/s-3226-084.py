def search(nums):
     n=len(nums)
     b=[]
     for x in nums:
        a=nums.count(x)
        if a>(n//2):
            if x not in b:
                b.append(x)
     if b!=[]:
            return b[0]
     elif b==[]:
            return "False"






nums = eval(input())
y = search(nums)
print(y)


