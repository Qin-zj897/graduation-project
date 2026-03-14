def search(n):
    for x in n:
        s=0
        for y in n:
            if x==y : s+=1
        if s>len(n)/2 : return x
    return False





nums = eval(input())
y = search(nums)
print(y)


