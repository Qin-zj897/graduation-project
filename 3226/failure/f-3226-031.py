def search(y):
    for x in y:
        if y.count(x)>len(y)//2:
            return x
        else:
        print("False")





nums = eval(input())
y = search(nums)
print(y)


