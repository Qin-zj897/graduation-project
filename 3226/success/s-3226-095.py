def search(a):
    for x in a:
        if a.count(x)>len(a)//2:
            return x
            break
        else:
            continue
    return 'False'


            
    





nums = eval(input())
y = search(nums)
print(y)


