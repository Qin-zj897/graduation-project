def search(x):
    m=1
    a=len(x)
    b=a//2
    for i in x:
        c=x.count(i)
        if c>b:
            return (i)
            m=0
            break
        if m:
            return ('False')





nums = eval(input())
y = search(nums)
print(y)


