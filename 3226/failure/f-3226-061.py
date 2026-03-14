def search(a):
    b=len(a)//2
    for x in a:
        if a.count(x)>b:
            return a.count(x)





nums = eval(input())
y = search(nums)
print(y)


