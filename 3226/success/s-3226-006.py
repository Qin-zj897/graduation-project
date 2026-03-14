def search(l):
    l=list(l)
    for x in l:
        if l.count(x)>len(l)/2:
            return x
        else:
            return False





nums = eval(input())
y = search(nums)
print(y)


