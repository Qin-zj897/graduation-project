def search(nums):
    a = len(nums)
    for i in nums:
        b = nums.count(i)
        c = 0
        if b> a/2:
            c = 1
            return i
    if c==0:
        return("False")
        
    
   





nums = eval(input())
y = search(nums)
print(y)


