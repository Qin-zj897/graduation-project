def search(list):
    b=0
    for x in list:
        n=list.count(x)
        if n>len(list)//2:
            a=x
            b+=1
        else:
            continue
    if b==0:
        return False
    else:
        return a





nums = eval(input())
y = search(nums)
print(y)


