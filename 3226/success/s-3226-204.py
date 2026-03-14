def search(ls):
    l=len(ls)
    for i in ls:
        c=ls.count(i)
        if c>(l/2):
            return(i)
        else:
            return "False"

    





nums = eval(input())
y = search(nums)
print(y)


