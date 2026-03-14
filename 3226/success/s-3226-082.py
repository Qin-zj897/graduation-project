def search(ls):
    ls2=[]
    for i in ls:
        b=ls.count(i)
        if b>len(ls)/2:
            ls2.append(i)
            return(ls2[0])
            break    
    if ls2==[]:return(False)





nums = eval(input())
y = search(nums)
print(y)


