def search(arg):
    k = 0
    for i in range(len(arg)):
        a = arg.count(arg[i])
        if a >= len(arg) / 2:
            k = 1
            return arg[i]
        else:
            continue
    if k==0:
        print('False')
    elif k==1:
        pass





nums = eval(input())
y = search(nums)
print(y)


