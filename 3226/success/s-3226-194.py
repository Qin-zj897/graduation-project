def search(nums):
    n=len(nums)//2
    a=list(set(nums))
    lit=[]
    for i in a:
        b=nums.count(i)
        if b >n:
            lit.append(i)
    if lit==[]:
        return False
    else:
        return(max(lit))





nums = eval(input())
y = search(nums)
print(y)


