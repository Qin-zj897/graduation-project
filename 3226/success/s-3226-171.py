def search(ls):
    for i in ls:
        if ls.count(i) > len(ls)//2:
            return i
        else:
            return 'False'





nums = eval(input())
y = search(nums)
print(y)


