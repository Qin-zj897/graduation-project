def search(x):
    for i in x:
        c = x.count(i)
        n = len(x)
        nn = n//2
        if c > nn:
            return i
        else:
            return False  





nums = eval(input())
y = search(nums)
print(y)


