def search(num):
    n=len(num)//2
    lit=[]
    for i in range(len(num)):
        a=num.count(num[i])
        if a>n:
            return num[i]
        else:
            return False





nums = eval(input())
y = search(nums)
print(y)


