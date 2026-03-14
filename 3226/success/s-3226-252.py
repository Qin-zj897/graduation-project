def search(a):
    for i in range(len(a)):
        if a.count(a[i])>(len(a)//2):
            c=a[i]
            return c
    return False





nums = eval(input())
y = search(nums)
print(y)


