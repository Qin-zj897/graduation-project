def search(nus):
    for i in range (len(nus)):
        if nus.count(i)>((len(nus))/2):
            return i
        else:
            continue
    return False





nums = eval(input())
y = search(nums)
print(y)


