def search(list):
    a=0
    n=[]
    for i in list:
        if list.count(i)>a:
            a=list.count(i)
    b=len(list)/2
    if a>b:
        return(i)
    else:
        return(False)





nums = eval(input())
y = search(nums)
print(y)


