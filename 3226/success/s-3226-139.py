def search(a):
    n=len(a)
    z=n//2
    for x in a:
        if a.count(x)>z:
            return x
        else:
         return False





nums = eval(input())
y = search(nums)
print(y)


