def search(n): 
    for i in n:
        i_count = n.count(i)
        if i_count>len(n)//2:
            return i
        else:
            return "False"





nums = eval(input())
y = search(nums)
print(y)


