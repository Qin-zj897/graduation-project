def search(a):
    for i in a:
        if a.count(i)>len(a)//2:
            return i
    else:
         return False





nums = eval(input())
y = search(nums)
print(y)


