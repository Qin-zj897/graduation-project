def search(ls):
    l=len(ls)
    for i in ls:
        c=ls.count(i)
        if c>int(l/2):
            return(i)
            break
    else:
        return "False"

    





nums = eval(input())
y = search(nums)
print(y)


