def search(a):
    b = []
    for x in a:
        if a.count(x)>len(a)//2:
            b.append(x)
        if b==[]:return False
        else:
             return  max(b)





nums = eval(input())
y = search(nums)
print(y)


