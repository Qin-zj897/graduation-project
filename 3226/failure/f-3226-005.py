def search(l):
    l=list(l)
    i=[x for x in l if l.count(x)>len(l)/2]
    for x in i:
        while i.count(x)>1:
            i.remove(x)
    if len(i)==0:
        return False
    return i





nums = eval(input())
y = search(nums)
print(y)


