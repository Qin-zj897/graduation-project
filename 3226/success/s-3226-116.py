def search(a):
    m=0
    for i in a:
        if a.count(i)>len(a)//2:
            m+=1
            return i
    if m ==0:
        return False          






nums = eval(input())
y = search(nums)
print(y)


