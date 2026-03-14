def search(nums):
    lit=''
    lit1=list(set(nums))
    for i in lit1:
        a=nums.count(i)
        if a>(len(nums)//2):
            lit+=str(i)
    if lit=='':
        return False
    else:
        return(lit)





nums = eval(input())
y = search(nums)
print(y)


