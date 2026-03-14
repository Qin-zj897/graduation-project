def search(nums):
    m=[]
    for i in nums:
        if nums.count(i)>len(nums)//2:
            m.append(i)
            return i
        else:
            continue 
    if m==[]:
        return "False"





nums = eval(input())
y = search(nums)
print(y)


