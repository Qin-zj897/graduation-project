def search(list):
    for i in list:
        if list.count(i) > len(list)//2:
            return i
    
        return False






nums = eval(input())
y = search(nums)
print(y)


