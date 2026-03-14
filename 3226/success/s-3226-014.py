def search(n):
    i=0
    for x in n:
        a=n.count(x)
        if a>(len(n)//2):
            return(x)
        else:
            i+=1
    if i==len(n):
        return(False)
    else:
        pass





nums = eval(input())
y = search(nums)
print(y)


