def search(a):
n = len(a)
    for x in a:
        if a.count(x)>n//2:
            m = x
        else:
            m=False
    return m





nums = eval(input())
y = search(nums)
print(y)


